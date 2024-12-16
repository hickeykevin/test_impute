from lightning.pytorch import LightningModule
import torch
from torch.nn import functional as F
from pypots.classification.brits.core import _BRITS
from src.methods.brits.modules import MultiTaskBRITS
from torch.optim.adam import Adam
from torch.optim import Optimizer
from typing import Dict, List, Optional, Union


"""
The implementation of BRITS for the partially-observed time-series classification task.

Refer to the paper "Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li.
BRITS: Bidirectional recurrent imputation for time series.
In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018."

Notes
-----
Partial implementation uses code from https://github.com/caow13/BRITS.
The bugs in the original implementation are fixed here.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause



class BRITSLightningModule(LightningModule):
    """The PyTorch implementation of the BRITS model :cite:`cao2018BRITS`.

    Parameters
    ----------

    rnn_hidden_size :
        The size of the RNN hidden state.

    classification_weight :
        The loss weight for the classification task.

    reconstruction_weight :
        The loss weight for the reconstruction task.

    lr: 
        The learning rate for the optimizer.

    """

    def __init__(
        self,
        rnn_hidden_size: int,
        classification_weight: float = 1,
        reconstruction_weight: float = 1,
        lr: float = 1e-3,
        pypots: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
       
 
    def setup(self, stage: Optional[str] = "fit"):
        data_info = self.trainer.datamodule.data_info
        assert bool([x in data_info.keys() for x in ['n_time_steps', 'n_features', 'n_classes']]), "data_info should contain 'n_steps', 'n_features' and 'n_classes"
        n_steps = data_info["n_time_steps"]
        n_features = data_info["n_features"]
        n_classes = data_info["n_classes"]
        init_args = {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_classes": n_classes,
            "rnn_hidden_size": self.hparams.rnn_hidden_size,
            "classification_weight": self.hparams.classification_weight,
            "reconstruction_weight": self.hparams.reconstruction_weight,
        }
        self.model = _BRITS(**init_args) if self.hparams.pypots else MultiTaskBRITS(**init_args)
            
    def forward(self, data: Dict, training: bool):
        """
        Forward pass of the model.

        Args:
            data (Dict): Input data for the forward pass.
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            The output of the model's forward pass.
        """
        return self.model(data)
    

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step on the given batch.

        Args:
            batch: The input batch for training.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the following items:
            - 'loss': The loss value calculated during the training step.
            - 'clf_logits': The raw classificatin logits, used to calculate classification metrics.
            - 'imputed_data': The imputed data, which is original data with imputed values.
            - 'X_ori': The original input data.
            - 'indicating_mask': The mask indicating missing values in the input data.
        """
        batch = self._assemble_input_for_training(batch)
        out = self.model(batch)
        loss = out['loss'] 
        # self.log("train/loss", loss)
        return {
            "loss": loss,
            "clf_logits": out['classification_pred'],
            "imputed_data": out['imputed_data'],
            "X_ori": batch["forward"]["X"],
            "indicating_mask": batch["forward"]["missing_mask"]
            }

    def on_train_batch_end(self, *args, **kwargs):
        pass
    
    def validation_step(self, batch, batch_idx):
        # check if batch is not an empty dict (datamodule.hparams.val_ratio = 0.0)
       
        data = self._assemble_input_for_validating(batch)
        out = self.model(data, training=False)
        predictions: Dict = out['classification_pred']
        return {
            "loss": out['loss'],
            "clf_logits": predictions,
            "imputed_data": out['imputed_data'],
            "X_ori": data["X_ori"],
            "indicating_mask": data["indicating_mask"]
        }

    def on_validation_batch_end(self, *args, **kwargs):
        pass
        
    def test_step(self, batch, batch_idx):
        """
        Perform a single testing step.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing the following keys:
            - "clf_logits": The predicted classification logits.
            - "imputed_data": The imputed data.
            - "X_ori": The original input data.
            - "indicating_mask": The indicating mask.

        """
        batch = self._assemble_input_for_validating(batch)
        out = self.model(batch, training=False)
        predictions = out['classification_pred']

        return {
            "clf_logits": predictions,
            "imputed_data": out['imputed_data'],
            "X_ori": batch["X_ori"],
            "indicating_mask": batch["indicating_mask"]
        }

    def configure_optimizers(self) -> Optimizer:
        return Adam(params=self.model.parameters(), lr=self.hparams.lr)
    
    def _assemble_input_for_training(self, data): 

        """
        Collate function for the BRITS dataloader.

        Args:
            data (List[Dict]): List of records containing time series data from BRITSDataFormat.

        Returns:
            Dict: A dictionary containing the collated data.

        Raises:
            AssertionError: If the required keys are not found in the input list.
        """
        # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
        for k, v in data.items():
            if k == 'label':
                data[k] = v.long()
            else:
                data[k] = v.type_as(next(iter(self.model.parameters())))
        final_dict = {
            'forward': {"X": data['X'], "missing_mask": data['missing_mask'], "deltas": data['deltas']}, #TODO: check if this is correct
            'backward': {"X": data['back_X'], "missing_mask": data['back_missing_mask'], "deltas": data['back_deltas']},
            'label': data['label']
        }
        return final_dict 
    
    def _assemble_input_for_validating(self, data: List):
        """
        Assembles the input dictionary for validating the model.

        Args:
            data (List): The input data for validation.

        Returns:
            dict: The assembled input dictionary containing the input data for validation.
        """
        if len(data) == 0:
            return {}
        else:
            out = self._assemble_input_for_training(data)
            X_ori = data['X_ori']
            indicating_mask =  data['indicating_mask']

            final_dict = {
                **out,
                "X_ori": X_ori,
                "indicating_mask": indicating_mask
            }
            return final_dict
    

