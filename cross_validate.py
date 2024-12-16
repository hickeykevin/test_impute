# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os import path
from typing import List, Dict, Optional, Tuple, Any, Tuple
from tqdm import trange, tqdm

from sklearn import model_selection
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Subset

from lightning.fabric import Fabric, seed_everything
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch import LightningModule, Callback, LightningDataModule

from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
from torchmetrics import Metric, MetricCollection

from src.methods.brits.lightningmodule import BRITSLightningModule
from src.utils import instantiate_callbacks, instantiate_loggers, get_pylogger, get_metric_value
from src.callbacks.cv_metrics import ScoreCrossValidationCallback

log = get_pylogger("cross_validate.py")

class DummyTrainer:
    def __init__(self, datamodule):
        self.datamodule = datamodule

@hydra.main(version_base=None, config_path="config", config_name="cv")
def main(cfg: DictConfig):
    metric_dict = run(cfg=cfg)
    metric_value = get_metric_value(metric_dict, cfg.get("optimized_metric"), log)
    return metric_value

def run(cfg: DictConfig) -> Tuple[dict, dict]:
    os.environ["HYDRA_FULL_ERROR"] = "1"

    seed_everything(cfg.seed)  # instead of torch.manual_seed(...)

    log.info("Instantiating loggers...")
    loggers: Optional[List[fabric.loggers.Logger]] = instantiate_loggers(cfg.get('logger'), log)

    log.info(f"Instantiating callbacks...")
    callbacks: Optional[List[pl.Callback]] = instantiate_callbacks(cfg.get('callbacks'), log)
    # callbacks = [ScoreCrossValidationCallback(num_classes=2, average='macro')]
    
    log.info(f"Instantiating fabric <{cfg.trainer._target_}>") # fabric class found in trainer config
    # fabric: Fabric = instantiate(cfg.trainer, callbacks=callbacksss)
    fabric = Fabric(callbacks=callbacks, loggers=loggers)
    fabric.launch()
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    with fabric.rank_zero_first(local=False): 
        dm: pl.LightningDataModule = instantiate(cfg.data)
        dm.prepare_data()
    dm.setup()
    assert hasattr(dm, "data_info"), "DataModule does not have data_info attribute!"
    
    # log.info(f"Instantiating model <{cfg.model.lightningmodule._target_}>")
    # model: pl.LightningModule = instantiate(cfg.model.lightningmodule)
    # model.trainer = DummyTrainer(dm)
    # model.setup(stage="fit")

    log.info(f"Beginning cross-validation...")
    metric_dict = cross_validate(
        fabric, 
        # model, 
        dm, 
        cfg
    )

    return metric_dict 

def cross_validate(
    fabric: Fabric,
    # model: LightningModule,
    datamodule: LightningDataModule,
    cfg: DictConfig,
    ) -> None:
    folds = cfg.folds
    kfold = model_selection.KFold(n_splits=folds, shuffle=True)

    # # initialize n_splits models and optimizers
    # models = [model for _ in range(kfold.n_splits)]
    # optimizers = [model.configure_optimizers() for model in models]

    # # fabric setup for models and optimizers
    # for i in range(kfold.n_splits):
    #     models[i], optimizers[i] = fabric.setup(models[i], optimizers[i])

    dataset: Dataset  = datamodule.dataset

    metric_fold_scores = {"f1": 0.0}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        if fold == 0:
            log.info(f"Instantiating model <{cfg.model.lightningmodule._target_}>")
        model = instantiate(cfg.model.lightningmodule)
        model: pl.LightningModule = instantiate(cfg.model.lightningmodule)
        model.trainer = DummyTrainer(datamodule)
        model.setup(stage="fit", data_init_info=datamodule.data_info())

        optimizer = model.configure_optimizers()

        model, optimizer = fabric.setup(model, optimizer)
        log.info(f"Working on fold {fold}")
        fabric.call("on_fit_start")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        bs = datamodule.hparams.batch_size
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, sampler=train_subsampler
        )
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=bs, sampler=val_subsampler
        )

        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

        # model, optimizer = models[fold], optimizers[fold]

        train(model, train_loader, optimizer, fabric, cfg, fold)

        val_loss = validate(model, val_loader, fabric, cfg, fold)
        
        fabric.print(f"Fold {fold + 1} validation loss: {val_loss}")

    fabric.call("on_fit_end")
    metric_fold_scores: Dict[str, torch.Tensor] = fabric._callbacks[0].fold_scores
    fabric.log_dict(metric_fold_scores)

    log.info(f"{folds}-cv f1 score: {metric_fold_scores[cfg.optimized_metric].item()}")

    return metric_fold_scores


def train(
    model: LightningModule, 
    data_loader: LightningDataModule, 
    optimizer: Optimizer, 
    fabric: Fabric, 
    cfg: DictConfig, 
    fold: int
    ) -> None:
    # TRAINING LOOP
    fabric.call("on_train_start")
    model.train()
    for epoch in range(0, cfg.epochs):
        fabric.call("on_train_epoch_start")
        epoch_loss = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch_idx, batch in progress_bar:
            fabric.call("on_train_batch_start", batch=batch, batch_idx=batch_idx)
            optimizer.zero_grad()
            output = model.training_step(batch=batch, batch_idx=batch_idx)
            loss = output['loss']
            fabric.log_dict({"train/loss": loss.item()})
            fabric.backward(loss)  # instead of loss.backward()

            optimizer.step()

            fabric.call("on_train_batch_end", pl_module=model, outputs=output, batch=batch, batch_idx=batch_idx)

            # update progress bar
            epoch_loss += loss.item()
         
        fabric.call("on_train_epoch_end")
        epoch_loss /= len(data_loader)
        fabric.log_dict({
            f"train/fold_{fold}_epoch_loss": epoch_loss
        })
        # progress_bar.set_postfix(epoch_loss=epoch_loss)

    fabric.call("on_train_end")
    

@torch.no_grad()
def validate(
    model: LightningModule, 
    data_loader: DataLoader, 
    fabric: Fabric, 
    cfg: DictConfig, 
    fold: int, 
    ) -> torch.Tensor:
    
    fabric.call("on_validation_start")
    model.eval()
    loss = 0
    fabric.call("on_validation_epoch_start")

    for batch_idx, batch in enumerate(data_loader):
        fabric.call("on_validation_batch_start", batch=batch, batch_idx=batch_idx)
        output = model.validation_step(batch=batch, batch_idx=batch_idx)
        loss += output['loss']
        fabric.log("val/loss", output['loss'])

        fabric.call("on_validation_batch_end", outputs=output, batch=batch)

    fabric.call("on_validation_epoch_end")
    fabric.call("on_validation_end")
    
    # all_gather is used to aggregate the value across processes
    fold_loss = fabric.all_gather(loss).sum() / len(data_loader.dataset)
        
    return fold_loss 


if __name__ == "__main__":
    main()


