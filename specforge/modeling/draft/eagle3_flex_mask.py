from torch.nn.attention.flex_attention import (
    or_masks,
    and_masks,
    create_block_mask,
    flex_attention,
)
import torch


def generate_eagle3_mask(seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, shift_left: int = 0):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx - shift_left >= kv_idx
        padding_mask = kv_idx < seq_lengths[b]
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask


    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_shift_left_{shift_left}"
    return mask_mod

def test_eagle3_flex_mask():
    B = 1
    H = 1
    S = 128 * 8
    D = 128
    Q_LEN = S
    KV_LEN = S * 3
    data_type = torch.bfloat16
    query = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
    key_cache = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=data_type, requires_grad=True)
    value_cache = torch.randn(B, H, KV_LEN, D, device="cuda", dtype=data_type, requires_grad=True)
    seq_lengths = torch.tensor([S], device="cuda", dtype=torch.int32)
    block_mask = create_block_mask(
        mask_mod=generate_eagle3_mask(seq_lengths=seq_lengths, Q_LEN=Q_LEN, KV_LEN=KV_LEN, shift_left=128 * 2),
        B=1, 
        H=1,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=query.device
    )

    dense_mask = block_mask.to_dense()
    print(block_mask.to_string())
    print(dense_mask)
    output = flex_attention(
        query, key_cache, value_cache, block_mask=block_mask)


if __name__ == "__main__":
    test_eagle3_flex_mask()
