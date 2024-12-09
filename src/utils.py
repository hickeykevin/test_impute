import torch

def move_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_to_device(v, device) for v in item]
    else:
        return item

def check_device(item):
    if isinstance(item, torch.Tensor):
        return item.device.type == 'cpu'
    elif isinstance(item, dict):
        return any(check_device(v) for v in item.values())
    elif isinstance(item, list):
        return any(check_device(v) for v in item)
    else:
        return False
    
from torchmetrics import Metric

class MeanRelativeError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("absolute_error", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float(), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if mask is not None:
            self.absolute_error += torch.sum(torch.abs(preds - target) * mask)
            self.total += torch.sum(mask)
        else:
            self.absolute_error += torch.sum(torch.abs(preds - target))
            self.total += target.numel()

    def compute(self):
        return self.absolute_error / self.total

from torchmetrics.regression import MeanSquaredError
class MeanSquaredError(MeanSquaredError):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if mask is not None:
            self.sum_squared_error += torch.sum((preds - target) ** 2 * mask)
            self.total += torch.sum(mask.long())
        else:
            super().update(preds, target)


from torch.autograd import Variable
from typing import Dict, List, Any
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
import logging
from omegaconf import DictConfig
import hydra

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
try:
    from lightning.pytorch.loggers.logger import Logger
except ImportError:
    from lighting.pytorch.loggers.logger import LightningLoggerBase
    Logger = LightningLoggerBase



def instantiate_callbacks(callbacks_cfg: DictConfig, log) -> List[Callback]:
    """Instantiates callbacks from config."""

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig, log) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger



def get_metric_value(metric_dict: dict, metric_name: str, log) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger


def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x

# @dataclass
# class ModelCheckpointArgs(Callback):
#     _target_: str = "lightning.pytorch.callbacks.ModelCheckpoint"
#     monitor: str = "val_loss"
#     mode: str = "min"
#     save_top_k: int = 1
#     dirpath: str = "lightning_logs/"
#     filename: str = "{epoch}-{val_loss:.2f}"
#     verbose: bool = False
#     save_last: bool = False
#     save_weights_only: bool = False
#     auto_insert_metric_name: bool = True
#     every_n_epochs: int = 1

# @dataclass
# class EarlyStoppingArgs(Callback):
#     _target_: str = "lightning.pytorch.callbacks.EarlyStopping"
#     monitor: str = "val_loss"
#     min_delta: float = 0.0
#     patience: int = 3
#     verbose: bool = False
#     mode: str = "min"

# @dataclass
# class CallbacksArgs:
#     defaults: List[Any] = field(default_factory=lambda: [])
#     model_checkpoint =  ModelCheckpointArgs
#     early_stopping = EarlyStoppingArgs