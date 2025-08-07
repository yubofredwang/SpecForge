import argparse
import hashlib
import os

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.logging import create_logger, validate_logger_args
from specforge.utils import (
    get_last_checkpoint,
    print_with_rank,
    rank_0_priority,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-data-split", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument("--attention-backend", type=str, default="sdpa")

    # resume
    parser.add_argument("--resume", action="store_true")

    # logging args
    parser.add_argument(
        "--logger-backend",
        type=str,
        choices=["wandb", "mlflow", "none"],
        default="none",
        help="Logging backend to use"
    )
    
    # wandb args (kept for backward compatibility)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (deprecated, use --logger-backend wandb)")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    
    # mlflow args
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default=None)
    parser.add_argument("--mlflow-run-name", type=str, default=None)

    args = parser.parse_args()
    
    # Handle backward compatibility: if --wandb is set, use wandb backend
    if args.wandb and args.logger_backend == "none":
        args.logger_backend = "wandb"

    return parser, args


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def main():
    # initialize
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank(f"Initialized distributed environment")

    # Validate logger arguments
    validate_logger_args(parser, args)

    # Initialize logger
    logger_kwargs = {}
    if args.logger_backend == "wandb":
        logger_kwargs = {
            "api_key": args.wandb_key,
            "enabled": True
        }
    elif args.logger_backend == "mlflow":
        logger_kwargs = {
            "tracking_uri": args.mlflow_tracking_uri,
            "enabled": True
        }
    
    logger = create_logger(args.logger_backend, **logger_kwargs)
    
    # Initialize logging if enabled
    if args.logger_backend != "none":
        project = args.wandb_project if args.logger_backend == "wandb" else args.mlflow_experiment
        name = args.wandb_name if args.logger_backend == "wandb" else args.mlflow_run_name
        logger.initialize(project=project, name=name)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    if args.tp_size > 1:
        # to avoid CPU RAM OOM, we directly init the model on CUDA
        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
            device="cuda",
        ).eval()
    else:
        target_model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
    print_with_rank(f"Initialized target model")
    # load model with resume
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(draft_model_last_checkpoint, attention_backend=args.attention_backend)
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(draft_model_config, attention_backend=args.attention_backend)
            .cuda()
            .to(torch.bfloat16)
        )
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank(f"Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # convert to dataloader
    cache_key = hashlib.md5(args.train_data_path.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    eval_dataset, eval_eagle3_dataset = None, None
    if args.eval_data_split is not None:
        assert args.eval_data_path is None, "eval_data_path must be None when eval_data_split is provided!"
        datasets = train_dataset.train_test_split(test_size=float(args.eval_data_split), seed=0)
        train_dataset, eval_dataset = datasets["train"], datasets["test"]

    if args.eval_data_path is not None:
        cache_key = hashlib.md5(args.eval_data_path.encode()).hexdigest()
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]

    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=4,
        shuffle=True,
        pad_to_multiple_of=128 if args.attention_backend == "flex_attention" else None,
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized train dataloader")

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank(f"Loaded vocab mapping")


    if eval_dataset is not None:
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            pad_to_multiple_of=128 if args.attention_backend == "flex_attention" else None,
            process_group=get_dp_group(),
        )
        print_with_rank(f"Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    eagle3_model = OnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized Eagle3 FSDP model")

    # build other components
    optimizer = torch.optim.AdamW(eagle3_model.parameters(), lr=args.learning_rate)
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )
    print_with_rank(f"Initialized optimizer and scheduler")

    # resume
    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)

            optimizer.load_state_dict(state["optimizer_state_dict"])
            print_on_rank0("Successfully loaded optimizer state_dict.")

            scheduler.load_state_dict(state["scheduler_state_dict"])
            print_on_rank0("Successfully loaded scheduler state_dict.")

            start_epoch = state["epoch"] + 1
            print_on_rank0(f"Resuming from epoch {start_epoch}")
        else:
            print_on_rank0(
                f"Warning: Checkpoint directory {draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
            )

    dist.barrier()

    # start running
    print_on_rank0(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.module.length)]
        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            ploss.backward()
            optimizer.step()
            scheduler.step()

            logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            logger.log_if_enabled(logdict)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = acc_i / dist.get_world_size()
            acc_i = acc_i.item()
            logger.log_if_enabled({f"train/epochacc_{i}": acc_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = loss_i / dist.get_world_size()
            loss_i = loss_i.item()
            logger.log_if_enabled({f"train/epochploss_{i}": loss_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )

        # run evaluation
        if eval_dataset is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                )
                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            for i in range(len(epoch_acces)):
                acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = acc_i / dist.get_world_size()
                acc_i = acc_i.item()

                logger.log_if_enabled({f"eval/epochacc_{i}": acc_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = loss_i / dist.get_world_size()
                loss_i = loss_i.item()

                logger.log_if_enabled({f"eval/epochploss_{i}": loss_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )

        if epoch % args.save_interval == 0:
            # Save the model
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()

            with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
                model_state_dict = eagle3_model.state_dict()
                state_to_save = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                draft_model_state_dict = {
                    k.replace("draft_model.", ""): v
                    for k, v in model_state_dict.items()
                    if "draft_model." in k
                }

                if dist.get_rank() == 0:
                    torch.save(
                        state_to_save,
                        os.path.join(epoch_output_dir, "training_state.pt"),
                    )
                    print_on_rank0(
                        f"Saved full training state to {epoch_output_dir}/training_state.pt"
                    )
                    draft_model.save_pretrained(
                        epoch_output_dir,
                        state_dict=draft_model_state_dict,
                    )
                    print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                dist.barrier()

    # Finish logging
    logger.finish()
    destroy_distributed()


if __name__ == "__main__":
    main()
