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
from typing import List, Dict, Optional, Tuple, Any
import torch
from lightning.fabric import Fabric, seed_everything
import lightning.pytorch as pl
from sklearn import model_selection
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.classification import Accuracy
from src.methods.brits.lightningmodule import BRITSLightningModule
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utils import instantiate_callbacks, instantiate_loggers, get_pylogger, get_metric_value
from src.callbacks.metrics import CVClassificationMetricsCallback

log = get_pylogger("train.py")

class DummyTrainer:
    def __init__(self, datamodule):
        self.datamodule = datamodule


def train_dataloader(model, data_loader, optimizer, fabric, epoch, hparams, fold):
    # TRAINING LOOP
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # NOTE: no need to call `.to(device)` on the data, target
        optimizer.zero_grad()
        output = model.training_step(batch=(data, target), batch_idx=batch_idx)
        loss = output['loss']
        fabric.backward(loss)  # instead of loss.backward()

        optimizer.step()
        if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)}"
                f" ({100.0 * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

        fabric.call("on_train_batch_end", model=model, outputs=output, batch=[data,target], batch_idx=batch_idx)
        if hparams.dry_run:
            break


def validate_dataloader(model, data_loader, fabric, hparams, fold, metrics_cb):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # NOTE: no need to call `.to(device)` on the data, target
            output = model.validation_step(batch=(data, target), batch_idx=batch_idx)
            loss += output['loss']

            # Accuracy with torchmetrics
            metrics_cb.on_validation_batch_end(trainer=None, model=model, outputs=output, batch=[data,target], batch_idx=batch_idx)

            if hparams.dry_run:
                break

    # all_gather is used to aggregate the value across processes
    loss = fabric.all_gather(loss).sum() / len(data_loader.dataset)
    scores: Dict = metrics_cb.on_validation_epoch_end(trainer=None, model=model)

    print(f"\nFor fold: {fold} Validation set: Average loss: {loss:.4f}, F1: {scores['f1_mean']:.4f}")
    return scores['f1_mean']


def cross_validate(
        fabric: Fabric,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        metrics_cb: pl.Callback,
        cfg: DictConfig,
):
    seed_everything(cfg.seed)  # instead of torch.manual_seed(...)

    # Loop over different folds (shuffle = False by default so reproducible)
    folds = cfg.folds
    kfold = model_selection.KFold(n_splits=folds)

    # initialize n_splits models and optimizers
    models = [model for _ in range(kfold.n_splits)]
    optimizers = [model.configure_optimizers() for model in models]
    dataset = datamodule.dataset

    # fabric setup for models and optimizers
    for i in range(kfold.n_splits):
        models[i], optimizers[i] = fabric.setup(models[i], optimizers[i])

    # loop over epochs
    for epoch in range(1, cfg.epochs + 1):
        # loop over folds
        epoch_f1 = 0
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset.data_with_missing)):
            print(f"Working on fold {fold}")

            # initialize dataloaders based on folds
            batch_size = datamodule.hparams.batch_size
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_ids))

            # set up dataloaders to move data to the correct device
            train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

            # get model and optimizer for the current fold
            model, optimizer = models[fold], optimizers[fold]

            # train and validate
            train_dataloader(model, train_loader, optimizer, fabric, epoch, cfg, fold)
            epoch_f1 += validate_dataloader(model, val_loader, fabric, cfg, fold, metrics_cb)
            metrics_cb.val_clf_metrics.reset()

        # log epoch metrics
        print(f"Epoch {epoch} - Average f1: {epoch_f1 / kfold.n_splits}")

        if cfg.dry_run:
            break

    # When using distributed training, use `fabric.save`
    # to ensure the current process is allowed to save a checkpoint
    if cfg.save_model:
        fabric.save(model.state_dict(), "test.pt")

@hydra.main(version_base=None, config_path="config", config_name="cv")
def run(cfg: DictConfig) -> Tuple[dict, dict]:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    dm: pl.LightningDataModule = instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()
    assert hasattr(dm, "data_info"), "DataModule does not have data_info attribute!"
    
    log.info(f"Instantiating model <{cfg.model.lightningmodule._target_}>")
    model: pl.LightningModule = instantiate(cfg.model.lightningmodule)
    model.trainer = DummyTrainer(dm)
    model.setup(stage="fit")

    log.info(f"Instantiating callbacks...")
    metrics_cb = CVClassificationMetricsCallback(boot_val=False)
    
    # log.info("Instantiating loggers...")
    # loggers: Optional[List[pl.loggers.Logger]] = instantiate_loggers(cfg.get('logger'), log)

    log.info(f"Instantiating fabric <{cfg.trainer._target_}>")
    fabric: Fabric = instantiate(cfg.trainer)

    log.info(f"Beginning cross-validation...")
    cross_validate(fabric, model, dm, metrics_cb, cfg)

if __name__ == "__main__":
    run()


