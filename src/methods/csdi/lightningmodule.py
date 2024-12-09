from lightning.pytorch import LightningModule
import torch
from torch import nn as nn
from .components import MultiTaskBackboneCSDI
from pypots.imputation.csdi.core import _CSDI
from typing import Dict
from functools import partial

class CSDILightningModule(_CSDI, LightningModule):
    def __init__(
            self,
            n_layers,
            n_features,
            n_heads,
            n_channels,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,
            lr
        ):
        super(CSDILightningModule, self).__init__(
            n_layers=n_layers,
            n_features=n_features,
            n_heads=n_heads,
            n_channels=n_channels,
            d_time_embedding=d_time_embedding,
            d_feature_embedding=d_feature_embedding,
            d_diffusion_embedding=d_diffusion_embedding,
            is_unconditional=is_unconditional,
            n_diffusion_steps=n_diffusion_steps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.save_hyperparameters()
       

    def setup(self, stage):
        n_classes = self.trainer.datamodule.data_info['n_classes']
        d_target = self.trainer.datamodule.data_info['n_features']
        # n_features = self.trainer.datamodule.data_info["n_features"]
        hparams = {k: v for k, v in self.hparams.items() if k not in ["lr", "n_features"]}
        self.backbone = MultiTaskBackboneCSDI(
            n_classes=n_classes,
            d_target=d_target,
           **hparams
            )

    def forward(self, data: Dict):
        observed_data, indicating_mask, cond_mask, observed_tp, target = self._assemble_input_for_training(data, self.hparams.n_diffusion_steps)
        side_info = self.get_side_info(observed_tp, cond_mask)
        return self.backbone(observed_data, indicating_mask, cond_mask, side_info, is_train=False)
        

    def training_step(self, batch, batch_idx):
        observed_data, indicating_mask, cond_mask, observed_tp, target = self._assemble_input_for_training(batch, self.hparams.n_diffusion_steps)
        side_info = self.get_side_info(observed_tp, cond_mask.transpose(1,2)) #???
        loss, clf_loss, residual, clf_out = self.backbone.calc_loss(
            observed_data, target, cond_mask, indicating_mask, side_info, is_train=True
        )
        total_loss = loss + clf_loss
        self.log("train_loss", total_loss)
        return {
            "loss": total_loss,
            "clf_logits": clf_out,
            "imputed_data": residual,
            "X_ori": observed_data,
            "indicating_mask": indicating_mask
        }
    
    def validation_step(self, batch, batch_idx):
        observed_data, indicating_mask, cond_mask, observed_tp, target = self._assemble_input_for_training(batch, self.hparams.n_diffusion_steps)
        side_info = self.get_side_info(observed_tp, cond_mask.transpose(1,2))
        loss, clf_loss, residual, clf_output = self.backbone.calc_loss_valid(
            observed_data, target, cond_mask, indicating_mask, side_info, is_train=False
        )
        total_loss = loss + clf_loss
        self.log("val_loss", total_loss)
        return {
            "loss": total_loss,
            "clf_logits": clf_output,
            "imputed_data": residual,
            "X_ori": observed_data,
            "indicating_mask": indicating_mask
        }
    
    def _assemble_input_for_training(self, batch: Dict, n_steps: int): 
        #TODO check with the original code's method, they do permutes here
        if "X_ori" in batch.keys():
            observed_data = batch["X_ori"]
        else:
            observed_data = batch["X"]
        
        indicating_mask = batch["indicating_mask"]
        cond_mask = batch["missing_mask"]
        observed_tp = torch.arange(0, observed_data.shape[1], dtype=torch.float32, device=observed_data.device).repeat(observed_data.shape[0]).view(observed_data.shape[0], -1)
        target = batch["label"]

        return observed_data, cond_mask, indicating_mask, observed_tp, target
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    

    