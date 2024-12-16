from typing import List, Dict, Optional, Tuple, Any
import hydra
from hydra.utils import instantiate
import lightning.pytorch as pl
import os
from torch import compile

from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    dm: pl.LightningDataModule = instantiate(cfg.data)
    assert hasattr(dm, "data_info"), "DataModule does not have data_info attribute!"
    
    log.info(f"Instantiating model <{cfg.model.lightningmodule._target_}>")
    model: pl.LightningModule = instantiate(cfg.model.lightningmodule)
    # model = compile(model)

    log.info(f"Instantiating callbacks...")
    callbacks: Optional[List[pl.Callback]] = instantiate_callbacks(cfg.get("callbacks"), log)      

    log.info("Instantiating loggers...")
    loggers: Optional[List[pl.loggers.Logger]] = instantiate_loggers(cfg.get('logger'), log)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info("Starting training...")
    trainer.fit(model=model, datamodule=dm)

    if cfg.test:
        log.info("Starting testing...")
        trainer.test(model=model, datamodule=dm, ckpt_path='best')
    
    log.info(f"Training finished. Retrieving metrics. Experiment details found at <{cfg.paths.output_dir}>")
    train_metrics = trainer.callback_metrics
    return train_metrics
    

if __name__ == "__main__":
    main()