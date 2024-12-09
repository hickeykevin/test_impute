from lightning.pytorch import LightningModule
import torch
from torch import nn
from .components import Model_brits_att

class RicardoLightningmodule(LightningModule):
    def __init__(
            self,
            rnn_hidden_size: int,
            lr: float = 1e-3
        ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = "fit"):
        n_series = self.trainer.datamodule.data_info['n_features']
        seq_len = self.trainer.datamodule.data_info['n_time_steps']
        self.model = Model_brits_att(
            rnn_name='LSTM', 
            rnn_hid_size=self.hparams.rnn_hidden_size,
            n_series=n_series, 
            seq_len=seq_len
            )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        batch = self.assemble_input_for_training(batch)
        ret = self(batch)
        loss = ret['loss']
        return {
            "loss": loss, 
            "clf_logits": ret['predictions'],
            "imputed_data": ret['imputations'],
            "X_ori": batch['forward']['evals'],
            "indicating_mask": batch['forward']['eval_masks']
        }
    
    def test_step(self, batch, batch_idx):
        batch = self.assemble_input_for_training(batch)
        ret = self(batch)
        return {
            "clf_logits": ret['predictions'],
            "imputed_data": ret['imputations'],
            "X_ori": batch['forward']['evals'],
            "indicating_mask": batch['forward']['eval_masks']
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def assemble_input_for_training(self, data): 

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
        if self.trainer.datamodule.hparams.ricardo:
            return data
        else:
            for k, v in data.items():
                if k == 'label':
                    data[k] = v.long()
                else:
                    data[k] = v.type_as(list(self.parameters())[0])
            final_dict = {
                'forward': {
                    "values": data['X'], 
                    "masks": data['missing_mask'], 
                    "deltas": data['deltas'], 
                    'evals': data['X_ori'], 
                    'eval_masks': data['indicating_mask']
                    },
                
                'backward': {
                    "values": data['back_X'], 
                    "masks": data['back_missing_mask'], 
                    "deltas": data['back_deltas'], 
                    'evals': torch.flip(data['X_ori'], dims=[1]),
                    'eval_masks': torch.flip(data['indicating_mask'], dims=[1])
                    },
                
                'label': data['label'],
                'is_train': data['is_train']
            }
            return final_dict