from typing import Dict, Optional, Tuple
from lightning.pytorch import LightningModule
import torch
from src.utils import get_pylogger
from pypots.utils.metrics import calc_mae
from lightning.pytorch.core.optimizer import Optimizer
import torch
import torch.nn as nn
from torch.nn import functional as F
from pypots.nn.modules.saits import SaitsLoss
from src.methods.saits.components import MultiTaskSATIS
logger = get_pylogger(__name__)


class SAITSLightningModule(LightningModule):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool = True,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        clf_weight: float = 0.5,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        if d_model != n_heads * d_k:
            logger.warning(
                "‼️ d_model must = n_heads * d_k, it should be divisible by n_heads "
                f"and the result should be equal to d_k, but got d_model={d_model}, n_heads={n_heads}, d_k={d_k}"
            )
            d_model = n_heads * d_k
            logger.warning(
                f"⚠️ d_model is reset to {d_model} = n_heads ({n_heads}) * d_k ({d_k})"
            )

    def setup(self, stage: Optional[str] = None):
        data_info = self.trainer.datamodule.data_info
        assert bool([x in data_info.keys() for x in ['n_time_steps', 'n_features', 'n_classes']]), "data_info should contain 'n_steps', 'n_features' and 'n_classes"
        n_steps = data_info['n_time_steps']
        n_features = data_info['n_features']
        n_classes = data_info['n_classes']

        self.model = MultiTaskSATIS(
            n_layers=self.hparams.n_layers,
            n_steps=n_steps,
            n_classes=n_classes,
            n_features=n_features,
            d_model=self.hparams.d_model,
            d_ffn=self.hparams.d_ffn,
            n_heads=self.hparams.n_heads,
            d_k=self.hparams.d_k,
            d_v=self.hparams.d_v,
            dropout=self.hparams.dropout,
            attn_dropout=self.hparams.attn_dropout,
            diagonal_attention_mask=self.hparams.diagonal_attention_mask,
            ORT_weight=self.hparams.ORT_weight,
            MIT_weight=self.hparams.MIT_weight,
        )
        self.clf_criterion = torch.nn.BCEWithLogitsLoss()
        self.saits_loss_func = SaitsLoss(self.hparams.ORT_weight, self.hparams.MIT_weight)
        self.criterion = calc_mae
       

    def forward(self, inputs: Dict, diagonal_attention_mask: bool, training: bool) -> torch.Tensor:
        return self.model(inputs, diagonal_attention_mask, training)

    
    def imputation_loss(self, inputs: Dict, X_tilde_1, X_tilde_2, X_tilde_3, masks):
        ORT_loss = 0
        ORT_loss += self.criterion(X_tilde_1, inputs['X'], masks)
        ORT_loss += self.criterion(X_tilde_2, inputs['X'], masks)
        ORT_loss += self.criterion(X_tilde_3, inputs['X'], masks)
        ORT_loss /= 3

        MIT_loss = self.criterion(
            X_tilde_3, inputs["X_ori"], inputs["indicating_mask"]
        )

        
        # `loss` is always the item for backward propagating to update the model
        loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss
        return loss, ORT_loss, MIT_loss

    def clf_loss(self, clf_output, target):
        if isinstance(self.clf_criterion, torch.nn.BCEWithLogitsLoss):
            target = torch.nn.functional.one_hot(target, num_classes=2)
            target = target.type_as(clf_output)
        else:
            target = target
        clf_loss = self.clf_criterion(clf_output, target)
        return clf_loss

    def training_step(self, batch: Tuple, batch_idx: int) -> Dict:
        batch = self._assemble_input_for_training(batch)
        out = self(batch, diagonal_attention_mask=True,  training=True)
        imp_loss = out['loss']
        imp_loss *= 1 - self.hparams.clf_weight
        clf_loss = self.hparams.clf_weight * self.clf_loss(out['clf_output'], batch['label'])
        loss = imp_loss + clf_loss

        self.log_dict(
            {
                'train/loss': loss,
                'train/imputation_loss': imp_loss,
                'train/clf_loss': clf_loss,
            }
        )
        return {
            "loss": loss,
            "imputed_data": out['imputed_data'],
            "X_ori": batch["X_ori"],
            "indicating_mask": batch["indicating_mask"],
            "clf_logits": out['clf_output'],
        }


    def validation_step(self, batch, batch_idx):

        batch = self._assemble_input_for_validating(batch)
        out = self(batch, diagonal_attention_mask=True,  training=True)
        imp_loss = out['loss']
        imp_loss *= 1 - self.hparams.clf_weight
        clf_loss = self.hparams.clf_weight * self.clf_loss(out['clf_output'], batch['label'])
        loss = imp_loss + clf_loss
        self.log_dict(
            {
                'val/loss': loss,
                'val/imputation_loss': imp_loss,
                'val/clf_loss': clf_loss,
            }
        )
        
        outputs = {
            "loss": loss, 
            "imputed_data": out['imputed_data'], 
            "X_ori": batch["X_ori"], 
            "indicating_mask": batch["indicating_mask"], 
            "clf_logits": out['clf_output'], 
            "label": batch["label"]
        }
        return outputs
    
    def test_step(self, batch, batch_dix):
        batch = self._assemble_input_for_validating(batch)
        out = self(batch, diagonal_attention_mask=True,  training=False)

        outputs = {
            "imputed_data": out['imputed_data'], 
            "X_ori": batch["X_ori"], 
            "indicating_mask": batch["indicating_mask"], 
            "clf_logits": out['clf_output'], 
            "label": batch["label"]
        }
        return outputs

    def _assemble_input_for_training(self, batch):
        # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
        for k, v in batch.items():
            if k == 'label':
                batch[k] = v.long()
            else:
                batch[k] = v.type_as(next(iter(self.model.parameters())))

        final_dict = {
            "X": batch["X"],
            "missing_mask": batch["missing_mask"],
            "X_ori": batch["X_ori"],
            "indicating_mask": batch["indicating_mask"],
            "label": batch["label"]
        }
           
        return final_dict
    
    def _assemble_input_for_validating(self, batch):
        return self._assemble_input_for_training(batch)
    
    def configure_optimizers(self) -> Optimizer:
        from torch.optim.adam import Adam
        return Adam(params=self.parameters(), lr=self.hparams.lr)
        
         
       
