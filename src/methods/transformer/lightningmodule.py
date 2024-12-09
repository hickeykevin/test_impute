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
from pypots.nn.modules.transformer import TransformerEncoder, PositionalEncoding

from src.utils import MeanRelativeError, MeanSquaredError
logger = get_pylogger(__name__)

"""
The implementation of Transformer for the partially-observed time-series imputation task.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause



class MTTransformer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        n_classes: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.embedding = nn.Linear(n_features * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_steps)
        self.encoder = TransformerEncoder(
            n_layers,
            d_model,
            d_ffn,
            n_heads,
            d_k,
            d_v,
            dropout,
            attn_dropout,
        )

        # Attention Clf layers
        self.output_projection = nn.Linear(d_model, n_features)
        self.self_attention = torch.nn.Linear(d_model, 1)
        
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # apply SAITS loss function to Transformer on the imputation task
        self.saits_loss_func = SaitsLoss(ORT_weight, MIT_weight)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, missing_mask = inputs["X"], inputs["missing_mask"]

        # apply the SAITS embedding strategy, concatenate X and missing mask for input
        input_X = torch.cat([X, missing_mask], dim=2) # this concats the original data with the missing mask

        # Transformer encoder processing
        input_X = self.embedding(input_X) # (bs, n_time_steps, d_model), this is the "token embedding"
        input_X = self.dropout(self.position_enc(input_X))
        enc_output, _ = self.encoder(input_X) # (bs, n_time_steps, d_model) attention output
        
        # project the representation from the d_model-dimensional space to the original data space for output
        reconstruction = self.output_projection(enc_output)

        # replace the observed part with values from X
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction # (bs, n_time_steps, n_features) shape

        # Clf self-attention layer
        attention_scores = self.self_attention(enc_output) # --> (batch_size, 40, 1) shape
        attention_scores = F.softmax(attention_scores, dim=1)
        attended_output = torch.sum(enc_output * attention_scores, dim=1) # --> (batch_size, d_model) shape

        # Add classification head
        clf_output = self.output_net(attended_output) # --> (batch_size, n_classes) shape
        # ensemble the results as a dictionary for return
        results = {
            "imputed_data": imputed_data,
            "learned_representation": reconstruction,
            "clf_output": clf_output
        }
        return results

class ClfTransformer(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        n_classes: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.embedding = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_enc = PositionalEncoding(d_model, n_positions=n_steps)
        self.encoder = TransformerEncoder(
            n_layers,
            d_model,
            d_ffn,
            n_heads,
            d_k,
            d_v,
            dropout,
            attn_dropout,
        )

        # Attention Clf layers
        self.output_projection = nn.Linear(d_model, n_features)
        self.self_attention = torch.nn.Linear(d_model, 1)
        
        self.output_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )


    def forward(self, inputs: dict) -> dict:
        X = inputs["X"]

        # apply the SAITS embedding strategy, concatenate X and missing mask for input

        # Transformer encoder processing
        X = self.embedding(X) # (bs, n_time_steps, d_model), this is the "token embedding"
        X = self.dropout(self.position_enc(X))
        enc_output, _ = self.encoder(X) # (bs, n_time_steps, d_model) attention output
    
        # Clf self-attention layer
        attention_scores = self.self_attention(enc_output) # --> (batch_size, 40, 1) shape
        attention_scores = F.softmax(attention_scores, dim=1)
        attended_output = torch.sum(enc_output * attention_scores, dim=1) # --> (batch_size, d_model) shape

        # Add classification head
        clf_output = self.output_net(attended_output) # --> (batch_size, n_classes) shape
        # ensemble the results as a dictionary for return
        results = {
            "clf_output": clf_output
        }
        return results


class BaseClfTransformerLightningModule(LightningModule):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        n_layers: int,
        dropout: float = 0,
        attn_dropout: float = 0,
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

    def _return_data_info(self, datamodule):
        data_info = datamodule.data_info
        assert bool([x in data_info.keys() for x in ['n_time_steps', 'n_features', 'n_classes']]), "data_info should contain 'n_steps', 'n_features' and 'n_classes"
        n_steps = data_info["n_time_steps"]
        n_features = data_info["n_features"]
        n_classes = data_info["n_classes"]
        return n_steps, n_features, n_classes

    def setup(self, stage: Optional[str] = None):
        n_steps, n_features, n_classes = self._return_data_info(self.trainer.datamodule)
        self.model = ClfTransformer(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=self.hparams.n_layers,
            n_classes=n_classes,
            d_model=self.hparams.d_model,
            d_ffn=self.hparams.d_ffn,
            n_heads=self.hparams.n_heads,
            d_k=self.hparams.d_k,
            d_v=self.hparams.d_v,
            dropout=self.hparams.dropout,
            attn_dropout=self.hparams.attn_dropout,
        )
        self.clf_criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()

    def forward(self, inputs: Dict):
        return self.model(inputs)
    
    def calculate_clf_loss(self, clf_output, target):
        if isinstance(self.clf_criterion, torch.nn.BCEWithLogitsLoss):
            target = torch.nn.functional.one_hot(target, num_classes=2)
            target = target.type_as(clf_output)
        else:
            target = target
        clf_loss = self.clf_criterion(clf_output, target)
        return clf_loss
    
    def training_step(self, batch, batch_idx):
        batch = self._assemble_input_for_training(batch)
        out = self(batch)
        clf_output = out['clf_output']
        clf_loss = self.calculate_clf_loss(clf_output, batch['label'])
        self.log("train/loss", clf_loss)
        return {
            "loss": clf_loss,
            "clf_logits": clf_output,
            "label": batch["label"]
        }

    def _assemble_input_for_training(self, batch):
        # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
        for k, v in batch.items():
            if k == 'label':
                batch[k] = v.long()
            else:
                batch[k] = v.type_as(next(iter(self.model.parameters())))

        final_dict = {
            "X": batch["X_ori"],
            "label": batch["label"]
        }
        return final_dict
    
    def _assemble_input_for_validating(self, batch):
        return self._assemble_input_for_training(batch)
    
    def shared_validation_step(self, batch, batch_idx):
    
        batch = self._assemble_input_for_validating(batch)
        out = self(batch)
        clf_output = out['clf_output']
        return clf_output
    
    def validation_step(self, batch, batch_idx):
        if not self.trainer.datamodule.hparams.val_ratio > 0:
            return
        clf_output = self.shared_validation_step(batch, batch_idx)
        
        return {
            "clf_logits": clf_output,
            "label": batch["label"]
        }
    
    def test_step(self, batch, batch_idx):
        clf_output: Dict = self.shared_validation_step(batch, batch_idx)
        outputs = {
            "clf_logits": clf_output['clf_output'],
            "label": batch["label"]
        }
        return outputs
    
    def configure_optimizers(self) -> Optimizer:
        from torch.optim.adam import Adam
        return Adam(params=self.model.parameters(), lr=self.hparams.lr)


    
class TransformerLightningModule(LightningModule):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        n_layers: int,
        dropout: float = 0,
        attn_dropout: float = 0,
        ORT_weight: int = 1,
        MIT_weight: int = 1,
        lr: float = 1e-3,
        clf_weight: float = 0.5
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
        n_steps = data_info["n_time_steps"]
        n_features = data_info["n_features"]
        n_classes = data_info["n_classes"]
        self.model = MTTransformer(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=self.hparams.n_layers,
            n_classes=n_classes,
            d_model=self.hparams.d_model,
            d_ffn=self.hparams.d_ffn,
            n_heads=self.hparams.n_heads,
            d_k=self.hparams.d_k,
            d_v=self.hparams.d_v,
            dropout=self.hparams.dropout,
            attn_dropout=self.hparams.attn_dropout,
            ORT_weight=self.hparams.ORT_weight,
            MIT_weight=self.hparams.MIT_weight,
        )

        self.clf_criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()
         # apply SAITS loss function to Transformer on the imputation task
        self.saits_loss_func = SaitsLoss(self.hparams.ORT_weight, self.hparams.MIT_weight)

    
    def forward(self, inputs: Dict):
        return self.model(inputs)
    
    def imputation_loss(self, reconstruction, X_ori, missing_mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        loss, ORT_loss, MIT_loss = self.saits_loss_func(
            reconstruction, X_ori, missing_mask, missing_mask
        )
        return loss, ORT_loss, MIT_loss
    
    def calculate_clf_loss(self, clf_output, target):
        if isinstance(self.clf_criterion, torch.nn.BCEWithLogitsLoss):
            target = torch.nn.functional.one_hot(target, num_classes=2)
            target = target.type_as(clf_output)
        else:
            target = target
        clf_loss = self.clf_criterion(clf_output, target)
        return clf_loss
    
    def training_step(self, batch, batch_idx):
        batch = self._assemble_input_for_training(batch)
        out = self(batch)
        imputed_data, learned_representation, clf_output = out['imputed_data'], out['learned_representation'], out['clf_output']    
        imputation_loss, ORT_loss, MIT_loss = self.imputation_loss(
            reconstruction=learned_representation, 
            X_ori=batch["X_ori"],
            missing_mask=batch["indicating_mask"],
        )
        imputation_loss *= 1-self.hparams.clf_weight
        clf_loss = self.hparams.clf_weight*self.calculate_clf_loss(clf_output, batch['label'])
        loss = imputation_loss + clf_loss
        
        self.log_dict({
            "train/loss": loss,
            "train/imputation_loss": imputation_loss,
            "train/clf_loss": clf_loss
        })
        
        return {
            "loss": loss,
            "imputed_data": imputed_data,
            "X_ori": batch["X_ori"],
            "indicating_mask": batch["indicating_mask"],
            "clf_logits": clf_output,
        }
    
    
    def shared_validation_step(self, batch, batch_idx):

        batch = self._assemble_input_for_validating(batch)
        out = self(batch)
        imputed_data, learned_representation, clf_output = out['imputed_data'], out['learned_representation'], out['clf_output']
        
        imputation_loss, *_ = self.imputation_loss(
            reconstruction=learned_representation, 
            X_ori=batch["X_ori"],
            missing_mask=batch["indicating_mask"],
        )
        clf_loss = self.calculate_clf_loss(clf_output, batch['label'])
        loss = imputation_loss + clf_loss
        
        outputs = {
            "loss": loss, 
            "imputed_data": imputed_data, 
            "X_ori": batch["X_ori"], 
            "indicating_mask": batch["indicating_mask"], 
            "clf_out": clf_output, 
            "label": batch["label"]
        }
        return outputs
    
    def validation_step(self, batch, batch_idx):
        if not self.trainer.datamodule.hparams.val_ratio > 0:
            return
        batch = self._assemble_input_for_validating(batch)
        outputs = self.shared_validation_step(batch, batch_idx)
        return {
            "imputed_data": outputs["imputed_data"],
            "X_ori": outputs["X_ori"],
            "indicating_mask": outputs["indicating_mask"],
            "clf_logits": outputs["clf_out"],
            "label": outputs["label"]
        }
    
    def test_step(self, batch, batch_idx):
        batch = self._assemble_input_for_validating(batch)
        outputs = self.shared_validation_step(batch, batch_idx)        
        return {
            "imputed_data": outputs["imputed_data"],
            "X_ori": outputs["X_ori"],
            "indicating_mask": outputs["indicating_mask"],
            "clf_logits": outputs["clf_out"],
            "label": outputs["label"]
        }
    

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
        return Adam(params=self.model.parameters(), lr=self.hparams.lr)
    
    # def validation_step(self, batch, batch_idx):
    #     outputs = self.shared_validation_step(batch, batch_idx, test=False)
    #     return outputs

    # def calculate_loss(self, learned_representation, inputs: Dict, training: bool = True):
    #     X, masks = inputs['X'], inputs['missing_mask']
    #     ORT_loss = calc_mae(learned_representation, X, masks)
    #     MIT_loss = calc_mae(
    #         learned_representation, inputs["X_ori"], inputs["indicating_mask"]
    #     )
    #     if training:
    #         # `loss` is always the item for backward propagating to update the model
    #         loss = self.hparams.ORT_weight * ORT_loss + self.hparams.MIT_weight * MIT_loss
    #         return loss
    #     else:
    #         return ORT_loss, MIT_loss
# class MTTransformerEncoder(_TransformerEncoder):
#     def __init__(self, n_classes, clf_dropout: float = 0.2, **kwargs):
#         super().__init__(**kwargs)
#         assert 'd_model' in kwargs.keys(), 'd_model should be provided in kwargs'
#         # self.self_attention = torch.nn.Linear(kwargs['d_model'], 1)
#         # self.clf_head = torch.nn.Sequential(
#         #     torch.nn.Linear(kwargs['d_model'], 128),
#         #     torch.nn.ReLU(),
#         #     torch.nn.Linear(128, n_classes)
#         # )
#         # Output classifier per sequence lement
#         self.output_net = nn.Sequential(
#             nn.Linear(kwargs['d_model'], kwargs['d_model']),
#             nn.LayerNorm(kwargs['d_model']),
#             nn.ReLU(inplace=True),
#             nn.Dropout(clf_dropout),
#             nn.Linear(kwargs['d_model'], n_classes),
#         )

#     def _process(self, inputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         X, masks, label = inputs["X"], inputs["missing_mask"], inputs['label'] # X is (batch_size, n_time_steps, n_features) tensor, masks is (batch_size, n_time_steps, n_features) tensor
#         input_X = torch.cat([X, masks], dim=2)
#         input_X = self.embedding(input_X) # --> (batch_size, n_time_steps, d_model) shape
#         enc_output = self.dropout(self.position_enc(input_X))

#         for encoder_layer in self.layer_stack:
#             enc_output, _ = encoder_layer(enc_output) # (batch_size, n_time_steps, d_model) shape

#         clf_output = self.output_net(enc_output) # --> (batch_size, n_time_steps, n_classes) shape

#         learned_presentation = self.reduce_dim(enc_output) # (batch_size, n_time_steps, 12) shape
#         imputed_data = (
#             masks * X + (1 - masks) * learned_presentation
#         )  # replace non-missing part with original data

#         # # Add self-attention layer
#         # attention_scores = self.self_attention(enc_output) # --> (batch_size, 40, 1) shape
#         # attention_scores = F.softmax(attention_scores, dim=1)
#         # attended_output = torch.sum(enc_output * attention_scores, dim=1) # --> (batch_size, d_model) shape

#         # # Add classification head
#         # clf_output = self.clf_head(attended_output) # --> (batch_size, n_classes) shape

#         return imputed_data, learned_presentation, clf_output