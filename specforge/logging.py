import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch.distributed as dist


class BaseLogger(ABC):
    """Abstract base class for logging backends."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and (dist.get_rank() == 0)
    
    @abstractmethod
    def initialize(self, project: Optional[str] = None, name: Optional[str] = None, **kwargs):
        """Initialize the logger with project and run name."""
        pass
    
    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish logging and cleanup."""
        pass
    
    def log_if_enabled(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics only if logger is enabled and on rank 0."""
        if self.enabled:
            self.log(metrics, step)


class WandbLogger(BaseLogger):
    """Weights & Biases logger implementation."""
    
    def __init__(self, api_key: Optional[str] = None, enabled: bool = True):
        super().__init__(enabled)
        self.api_key = api_key
        self.run = None
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                raise ImportError("wandb is required for WandbLogger. Install with: pip install wandb")
    
    def initialize(self, project: Optional[str] = None, name: Optional[str] = None, **kwargs):
        """Initialize wandb logging."""
        if not self.enabled:
            return
            
        if self.api_key:
            self.wandb.login(key=self.api_key)
        
        self.run = self.wandb.init(
            project=project,
            name=name,
            **kwargs
        )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if not self.enabled or self.run is None:
            return
        
        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run is not None:
            self.wandb.finish()


class MLflowLogger(BaseLogger):
    """MLflow logger implementation."""
    
    def __init__(self, tracking_uri: Optional[str] = None, enabled: bool = True):
        super().__init__(enabled)
        self.tracking_uri = tracking_uri
        self.run_id = None
        
        if self.enabled:
            try:
                import mlflow
                self.mlflow = mlflow
            except ImportError:
                raise ImportError("mlflow is required for MLflowLogger. Install with: pip install mlflow")
    
    def initialize(self, project: Optional[str] = None, name: Optional[str] = None, **kwargs):
        """Initialize MLflow logging."""
        if not self.enabled:
            return
        
        if self.tracking_uri:
            self.mlflow.set_tracking_uri(self.tracking_uri)
        
        if project:
            self.mlflow.set_experiment(project)
        
        run = self.mlflow.start_run(run_name=name)
        self.run_id = run.info.run_id
        
        # Log any additional parameters
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                self.mlflow.log_param(key, value)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not self.enabled or self.run_id is None:
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.mlflow.log_metric(key, value, step=step)
    
    def finish(self):
        """Finish MLflow run."""
        if self.enabled and self.run_id is not None:
            self.mlflow.end_run()


class NoOpLogger(BaseLogger):
    """No-operation logger that does nothing."""
    
    def __init__(self):
        super().__init__(enabled=False)
    
    def initialize(self, project: Optional[str] = None, name: Optional[str] = None, **kwargs):
        pass
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass
    
    def finish(self):
        pass


def create_logger(backend: str, **kwargs) -> BaseLogger:
    """Factory function to create appropriate logger based on backend."""
    if backend == "wandb":
        return WandbLogger(**kwargs)
    elif backend == "mlflow":
        return MLflowLogger(**kwargs)
    elif backend == "none" or backend is None:
        return NoOpLogger()
    else:
        raise ValueError(f"Unknown logging backend: {backend}. Supported backends: wandb, mlflow, none")


def validate_logger_args(parser, args):
    """Validate logger arguments similar to the existing wandb validation."""
    if args.logger_backend == "none" or args.logger_backend is None:
        return
    
    if args.logger_backend == "wandb":
        validate_wandb_args(parser, args)
    elif args.logger_backend == "mlflow":
        # MLflow validation - tracking URI can be set via environment variable
        if args.mlflow_tracking_uri is None and "MLFLOW_TRACKING_URI" in os.environ:
            args.mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
