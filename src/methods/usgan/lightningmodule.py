"""
The implementation of USGAN for the partially-observed time-series imputation task.

Refer to the paper "Xiaoye Miao, Yangyang Wu, Jun Wang, Yunjun Gao, Xudong Mao, and Jianwei Yin.
Generative Semi-supervised Learning for Multivariate Time Series Imputation.
In AAAI, 35(10):8983–8991, May 2021."

"""

# Created by Jun Wang <jwangfx@connect.ust.hk> and Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from pypots.nn.modules.usgan.layers import UsganDiscriminator as Discriminator
import torch 
from src.methods.brits.modules import MultiTaskBRITS
from pypots.utils.metrics import calc_mse, calc_mae
import lightning.pytorch as pl
from torch import nn
from typing import Dict, Iterable, List, Union, Optional
from torch.optim import Optimizer, Adam
import torch.nn.functional as F



# # largely copied from pypots.imputation.usgan.model.py
# class MyUSGAN(nn.Module):
#     """The PyTorch implementation of the USGAN model. Refer to :cite:`miao2021SSGAN`.

#     Parameters
#     ----------
#     n_steps : int
#         The number of time steps in the time-series data sample.

#     n_features : int
#         The number of features in the time-series data sample.

#     rnn_hidden_size : int
#         The hidden size of the RNN cell

#     lambda_mse : float
#         The weight of the reconstruction loss

#     hint_rate : float
#         The hint rate for the discriminator

#     dropout : float
#         The dropout rate for the last layer in Discriminator

#     G_steps : int
#         The number of steps to train the generator in each iteration.

#     D_steps : int
#         The number of steps to train the discriminator in each iteration.

#     G_optimizer : :class:`pypots.optim.Optimizer`
#         The optimizer for the generator training.
#         If not given, will use a default Adam optimizer.

#     D_optimizer : :class:`pypots.optim.Optimizer`
#         The optimizer for the discriminator training.
#         If not given, will use a default Adam optimizer.

#     """

#     def __init__(
#         self,
#         n_steps: int,
#         n_features: int,
#         n_classes: int,
#         rnn_hidden_size: int,
#         lambda_mse: float,
#         hint_rate: float = 0.7,
#         dropout_rate: float = 0.0,
#         device: Union[str, torch.device] = "cpu", 
#         classification_weight: float = 0.5,
#         reconstruction_weight: float = 0.5,
#     ):
#         super().__init__()
#         self.generator = MultiTaskBRITS(
#             n_steps=n_steps, 
#             n_features=n_features, 
#             rnn_hidden_size=rnn_hidden_size, 
#             n_classes=n_classes,
#             classification_weight=classification_weight, # dummy, it is not used
#             reconstruction_weight=reconstruction_weight # dummy, it is not used
#         )
#         self.discriminator = Discriminator(
#             n_features,
#             rnn_hidden_size,
#             hint_rate=hint_rate,
#             dropout_rate=dropout_rate,
#         )

#         self.lambda_mse = lambda_mse
#         self.device = device
        
        
    
#     # this is assuming data is formatted in the dictionary
#     def forward(
#         self,
#         inputs: dict,
#         training_object: str = "generator",
#         training: bool = True,
#     ) -> dict:
#         assert training_object in [
#             "generator",
#             "discriminator",
#         ], 'training_object should be "generator" or "discriminator"'

#         results = self.generator(inputs, training=training)
        
#         # if in training mode, return results with losses
#         if training:
#             forward_X = inputs["forward"]["X"]
#             forward_missing_mask = inputs["forward"]["missing_mask"]
#             imputed_data = results["imputed_data"]

#             if training_object == "discriminator":
#                 inputs["discrimination"] = self.discriminator(
#                     imputed_data.detach(), forward_missing_mask
#                 )
#                 l_D = F.binary_cross_entropy_with_logits(
#                     inputs["discrimination"], forward_missing_mask
#                 )
#                 results["discrimination_loss"] = l_D
#             else:
#                 inputs["discrimination"] = self.discriminator(
#                     imputed_data, forward_missing_mask
#                 )
#                 # discriminator loss component
#                 l_G = -F.binary_cross_entropy_with_logits(
#                     inputs["discrimination"],
#                     forward_missing_mask,
#                     weight=1 - forward_missing_mask,
#                 )

                
#                 # reconstructed loss is the 3 terms in equation (2) in paper
#                 # but is modified to be mse of reconstructed_data which is the 
#                 # average of forward rits and backward rits imputed data
#                 # the last term is the consistency loss 
#                 reconstruction_loss = calc_mse(
#                     forward_X, results["reconstructed_data"], forward_missing_mask
#                 ) + 0.1 * calc_mse(
#                      results["f_reconstruction"],  results["b_reconstruction"]
#                 )
#                 y_h_loss = results['y_h_loss']
#                 loss_gene = l_G + self.lambda_mse * reconstruction_loss + y_h_loss
#                 results["generation_loss"] = loss_gene

#         return results



class USGANLightningModule(pl.LightningModule):
    def __init__(
            self,
            rnn_hidden_size: int,
            lambda_mse: float = 1,
            hint_rate: float = 0.7,
            dropout: float = 0.0,
            G_steps: int = 1,
            D_steps: int = 1,
            G_lr: float = 0.01,
            D_lr: float = 0.01,
            classification_weight: float = 0.5,
            reconstruction_weight: float = 0.5,
            **kwargs
     ):
        super().__init__()
        self.save_hyperparameters()
        assert G_steps > 0 and D_steps > 0, "G_steps and D_steps should both >0"
        self.automatic_optimization = False

    def setup(self, stage: Optional[str] = 'fit'):
        data_info = self.trainer.datamodule.data_info
        assert bool([x in data_info.keys() for x in ['n_time_steps', 'n_features']]), "data_info should contain 'n_steps' and 'n_features'"
        n_steps = data_info['n_time_steps']
        n_features = data_info['n_features']
        n_classes = data_info['n_classes']
        
        self.generator = MultiTaskBRITS(
            n_steps=n_steps, 
            n_features=n_features, 
            rnn_hidden_size=self.hparams.rnn_hidden_size, 
            n_classes=n_classes,
            classification_weight=self.hparams.classification_weight, # dummy, it is not used
            reconstruction_weight=self.hparams.reconstruction_weight # dummy, it is not used
        )
        self.discriminator = Discriminator(
            n_features=n_features,
            rnn_hidden_size=self.hparams.rnn_hidden_size,
            hint_rate=self.hparams.hint_rate,
            dropout_rate=self.hparams.dropout
        )
        
    
    def generator_loss(self, gen_out, forward_X, forward_missing_mask, labels):

        disc_out = self.discriminator_output(
            gen_out['imputed_data'], forward_missing_mask, train_disc=False
        )
        
        l_G = -F.binary_cross_entropy_with_logits(
            disc_out,
            forward_missing_mask,
            weight=1-forward_missing_mask,
        )
        
        reconstruction_loss = calc_mse(
            forward_X, gen_out["reconstruction"], forward_missing_mask
        ) + 0.1 * calc_mse(
            gen_out["f_reconstruction"],  gen_out["b_reconstruction"]
        )
        
        clf_loss = gen_out["classification_loss"] # is returned by MultiTaskBRITS forward 

        losses = l_G + self.hparams.lambda_mse, reconstruction_loss, clf_loss
        return losses
    
    def discriminator_output(self, imputed_data, forward_missing_mask, train_disc: bool=False):
        if train_disc:
            disc_out = self.discriminator(
                    imputed_data.detach(), forward_missing_mask
                )
            l_D = F.binary_cross_entropy_with_logits(
                    disc_out, forward_missing_mask
                )
            return l_D
        else:
            disc_out = self.discriminator(
                    imputed_data, forward_missing_mask
                )
            return disc_out
        
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        batch = self._assemble_input(batch)
        gen_out = self.generator(batch, training=True)
        
        if batch_idx % self.hparams.G_steps == 0:
            opt_g.zero_grad()
            l_G, reconstruction_loss, clf_loss = self.generator_loss(
                gen_out, 
                batch["forward"]["X"], 
                batch["forward"]["missing_mask"],
                batch["label"]
            )
            total_loss = l_G + reconstruction_loss + clf_loss
            self.manual_backward(total_loss)
            opt_g.step()
            
            self.log(f"train/imputation_loss", reconstruction_loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train/clf_loss", clf_loss, on_step=True, on_epoch=True, sync_dist=True)

        if batch_idx % self.hparams.D_steps == 0:
            opt_d.zero_grad()
            l_D = self.discriminator_output(gen_out['imputed_data'], batch["forward"]["missing_mask"], train_disc=True)
            self.manual_backward(l_D)
            opt_d.step()
            self.log("train/discrimination_loss", l_D, on_step=True, on_epoch=True, sync_dist=True)

        return {
            "clf_logits": gen_out["classification_pred"], #these are softmax probabilities technically
            "imputed_data": gen_out["imputed_data"],
            "X_ori": batch['X_ori'],
            "indicating_mask": batch['indicating_mask']
        }
   
    def validation_step(self, batch, batch_idx):
        if not self.trainer.datamodule.hparams.val_ratio > 0:
            return
        batch = self._assemble_input(batch)
        gen_out = self.generator(batch, training=False)
        step_artifacts = {
            "clf_logits": gen_out["classification_pred"],
            "imputed_data": gen_out["imputed_data"],
            "X_ori": batch["X_ori"],
            "indicating_mask": batch["indicating_mask"],
        }
        
        return step_artifacts
    
    def test_step(self, batch, batch_idx):
        batch = self._assemble_input(batch)
        gen_out = self.generator(batch, training=False)
        step_artifacts = {
            "clf_logits": gen_out["classification_pred"],
            "imputed_data": gen_out["imputed_data"],
            "X_ori": batch["X_ori"],
            "indicating_mask": batch["indicating_mask"],
        }
        
        return step_artifacts

                
    def configure_optimizers(self):
        G_optimizer = Adam(params=self.generator.parameters(), lr=self.hparams.G_lr)
        D_optimizer = Adam(params=self.discriminator.parameters(), lr=self.hparams.D_lr)
        return G_optimizer, D_optimizer
    
    def _assemble_input(self, data): 

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
                data[k] = v.type_as(next(iter(self.generator.parameters())))
        final_dict = {
            'forward': {"X": data['X'], "missing_mask": data['missing_mask'], "deltas": data['deltas']}, #TODO: check if this is correct
            'backward': {"X": data['back_X'], "missing_mask": data['back_missing_mask'], "deltas": data['back_deltas']},
            'label': data['label'],
            'X_ori': data['X_ori'],
            'indicating_mask': data['indicating_mask']
        }
        return final_dict 
    
    # def _assemble_input_for_validating(self, data: Dict):
    #     out = self._assemble_input_for_training(data)
    #     X_ori = data['X_ori']
    #     indicating_mask =  data['indicating_mask']

    #     final_dict = {
    #         **out,
    #         "X_ori": X_ori,
    #         "indicating_mask": indicating_mask
    #     }
    #     return final_dict
        
    
    
        

