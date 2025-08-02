from abc import ABC, abstractmethod

# Logging abstraction
class BaseLogger(ABC):
    
    @abstractmethod
    def init(self, **kwargs):
        pass
    
    @abstractmethod
    def log(self, log_dict):
        pass
    
    @abstractmethod
    def is_initialized(self):
        pass


class WandbLogger(BaseLogger):
    
    def __init__(self):
        self._initialized = False
    
    def init(self, project=None, name=None, key=None, **kwargs):
        if key:
            wandb.login(key=key)
        wandb.init(project=project, name=name)
        self._initialized = True
    
    def log(self, log_dict):
        if self.is_initialized():
            wandb.log(log_dict)
    
    def is_initialized(self):
        return self._initialized and wandb.run is not None


class MLflowLogger(BaseLogger):
    
    def __init__(self):
        self._initialized = False
        self._mlflow = None
    
    def init(self, experiment_name=None, run_name=None, tracking_uri=None, **kwargs):
        try:
            import mlflow
            self._mlflow = mlflow
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            mlflow.start_run(run_name=run_name)
            self._initialized = True
        except ImportError:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")
    
    def log(self, log_dict):
        if self.is_initialized():
            for key, value in log_dict.items():
                self._mlflow.log_metric(key, value)
    
    def is_initialized(self):
        return self._initialized and self._mlflow.active_run() is not None


class NoOpLogger(BaseLogger):

    def init(self, **kwargs):
        pass
    
    def log(self, log_dict):
        pass
    
    def is_initialized(self):
        return False


def create_logger(logger_type, wandb=False):
    if logger_type == "wandb" or wandb:
        return WandbLogger()
    elif logger_type == "mlflow":
        return MLflowLogger()
    elif logger_type == "none":
        return NoOpLogger()
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
