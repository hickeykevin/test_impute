from pypots.classification.brits import BRITS
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


class BaseBRITSLightningModule(LightningModule):
    def __init__(self, lr: float = 1e-3,**kwargs):
        """
        Initializes the LightningModule with the specified learning rate and additional keyword arguments.
        Args:
            lr (float): Learning rate for the optimizer. Default is 1e-3.
            **kwargs: Additional keyword for pypots.classification.brits.BRITS.
        
        **kwargs: Additional keyword arguments for pypots.classification.brits.BRITS.
            n_steps: The number of time steps in the time-series data sample.
            n_features: The number of features in the time-series data sample.
            n_classes: The number of classes in the classification task.
            rnn_hidden_size: The size of the RNN hidden state.
            classification_weight: The loss weight for the classification task.
            reconstruction_weight: The loss weight for the reconstruction task.
        """
        
        super().__init__()
        self.save_hyperparameters()
        self.model = None #instantiated in setup()
        
    def configure_optimizers(self) -> Optimizer:
        return Adam(params=self.model.parameters(), lr=self.hparams.lr)
    
    def _set_data_info(self, data_info: Union[None, Dict[str, Union[int, float]]] = None) -> Dict[str, Union[int, float]]:
        """
        Sets the data information for the model.
        This method initializes the data information required for the model, such as the number of time steps,
        features, and classes. If `data_info` is not provided, it retrieves the information from the trainer's
        datamodule.
        Args:
            data_info (Union[None, Dict[str, Union[int, float]]], optional): A dictionary containing data information
                with keys 'n_time_steps', 'n_features', and 'n_classes'. If None, the information is retrieved from
                the trainer's datamodule.
        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the initialized arguments for the model, including
                'n_steps', 'n_features', 'n_classes', 'rnn_hidden_size', 'classification_weight', and 'reconstruction_weight'.
        Raises:
            AssertionError: If `data_info` does not contain the required keys 'n_time_steps', 'n_features', and 'n_classes'.
        """

        if data_info is None: # assume using Trainer and not Fabric
            data_info: Dict[str, Union[int, float]] = self.trainer.datamodule.data_info
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
        return init_args

    def setup(self, stage: Optional[str], data_init_info: Dict[str, int] = None) -> None:
        if stage == 'fit':
            init_args = self._set_data_info(data_init_info)
            self.model = BRITS(**init_args).model

    def forward(self, data: Dict, training: bool):
        """
        Forward pass of the model.

        Args:
            data (Dict): Input data for the forward pass.
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            The output of the model's forward pass.
        """
        return self.model(data, training)
    
    def on_train_batch_end(self, *args, **kwargs):
        pass

    def training_step(self, batch: Dict[str: torch.Tensor], batch_idx):
        """
        Performs a single training step on the given batch.

        Args:
            batch (Dict): The input batch for training.
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
        out = self.model(batch, training=True)
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
    
    def validation_step(self, batch: Dict[str: torch.Tensor], batch_idx):
        """
        Perform a single validation step.
        Args:
            batch (Dict): The input batch of data.
            batch_idx (int): The index of the batch.
        Returns:
            dict: A dictionary containing the classification logits and the loss.
                - "clf_logits" (Dict): The classification predictions.
                - "loss" (torch.Tensor): The computed loss.
        """

        data = self._assemble_input_for_validating(batch)
        out = self.model(data, training=True) #allow to return loss
        predictions: Dict = out['classification_pred']
        loss: torch.Tensor = out['loss']
        return {
            "clf_logits": predictions,
            "loss": loss,
        }

    def on_validation_batch_end(self, *args, **kwargs):
        pass
        
    def on_test_batch_end(self, *args, **kwargs):
        pass
    
    def test_step(self, batch: Dict[str: torch.Tensor], batch_idx):
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
    
    def on_test_epoch_end(self, *args, **kwargs):
        pass
    
    def _assemble_input_for_training(self, data): 

        """
        Collate function for the BRITS Datamodule DataLoader.

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

class BRITSLightningModule(BaseBRITSLightningModule):

    def __init__(self, lr: float = 1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
       
 
    def setup(self, stage: Optional[str], data_init_info: Union[None, Dict[str, Union[int, float]]] = None) -> None:
        if stage == "fit":
            init_args = self._set_data_info(data_init_info)
            self.model = MultiTaskBRITS(**init_args)

    def configure_optimizers(self) -> Optimizer:
        return Adam(params=self.model.parameters(), lr=self.hparams.lr)
    
            
class BRITSLightningModuleWrapper(LightningModule):
    def __init__(self, missing_ratio: float, **kwargs):
        """
        Initializes the associated BRITS LightningModule based on the missing ratio.
        Args:
            missing_ratio (float): The ratio of missing data. If 0.0, initializes BaseBRITSLightningModule.
            **kwargs: Additional keyword arguments passed to the model initialization.
        """

        super().__init__()
        if missing_ratio == 0.0:
            self.model = BaseBRITSLightningModule(**kwargs)
        else:
            self.model = BRITSLightningModule(**kwargs)

    def forward(self, data: Dict, training: bool):
        return self.model(data, training)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def setup(self, stage: Optional[str], data_init_info: Dict[str, int] = None):
        self.model.setup(stage, data_init_info)

    def on_train_batch_start(self, *args, **kwargs):
        return self.model.on_train_batch_start(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def on_train_batch_end(self, *args, **kwargs):
        return self.model.on_train_batch_end(*args, **kwargs)
    
    def on_validation_batch_end(self, *args, **kwargs):
        return self.model.on_validation_batch_end(*args, **kwargs)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def on_test_batch_start(self, *args, **kwargs):
        return self.model.on_test_batch_start(*args, **kwargs)
    
    def test_step(self, batch, batch_idx):
        return self.model.test_step(batch, batch_idx)
    
    def on_test_batch_end(self, *args, **kwargs):
        return self.model.on_test_batch_end(*args, **kwargs)
    
    
    
        
    


