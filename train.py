from typing import List, Dict, Optional, Tuple, Any
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from src.utils import instantiate_callbacks, instantiate_loggers, get_pylogger, get_metric_value
import lightning.pytorch as pl
import os
from torch import compile

log = get_pylogger("train.py")


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    # import pdb; pdb.set_trace()
    metric_dict = run(cfg=cfg)
    metric_value = get_metric_value(metric_dict, cfg.get("optimized_metric"), log)
    return metric_value


def run(cfg: DictConfig) -> Tuple[dict, dict]:
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