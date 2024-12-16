from typing import Dict, Tuple
from torch import nn
import torch
from pypots.classification.brits.core import _BRITS
from pypots.utils.metrics.error import calc_mae, calc_mse
import torch.nn.functional as F
from pypots.nn.modules.brits.backbone import BackboneRITS, BackboneBRITS

class MyBackboneRITS(BackboneRITS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # clf components
        self.W_s1 = nn.Linear(kwargs['rnn_hidden_size'], 350)
        self.W_s2 = nn.Linear(350, 30)
        self.out = nn.Linear(kwargs['rnn_hidden_size']*30, 1)
    
    def attention_rnn(self, rnn_output: torch.Tensor) -> torch.Tensor:
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(rnn_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def forward(
        self, inputs: dict, direction: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from `inputs`.

        Returns
        -------
        imputed_data :
            Input data with missing parts imputed. Shape of [batch size, sequence length, feature number].

        estimations :
            Reconstructed data. Shape of [batch size, sequence length, feature number].

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        X = inputs[direction]["X"]  # feature values
        missing_mask = inputs[direction]["missing_mask"]  # mask marks missing part in X
        deltas = inputs[direction]["deltas"]  # time-gap values

        device = X.device

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        cell_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        all_hidden_states = torch.zeros((X.size(0), X.size(1), self.rnn_hidden_size)).type_as(X)

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(device)

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = X[:, t, :]  # values
            m = missing_mask[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += calc_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += calc_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += calc_mae(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )
            all_hidden_states[:, t, :] = hidden_states

        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        reconstruction = torch.cat(estimations, dim=1)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction # real data at non-missing indicies, direction imputed data at missing indicies

        atten_weight_matrix = self.attention_rnn(all_hidden_states)
        hidden_matrix = torch.bmm(atten_weight_matrix, all_hidden_states)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        pred = self.out(attention_output)
        return {
            "imputed_data": imputed_data,
            "reconstruction": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "prediction": pred
        }
        # return imputed_data, reconstruction, hidden_states, all_hidden_states, reconstruction_loss
    
class MyBackboneBRITS(BackboneBRITS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rits_f = MyBackboneRITS(
            n_steps=kwargs['n_steps'], 
            n_features=kwargs['n_features'], 
            rnn_hidden_size=kwargs['rnn_hidden_size']
            )
        self.rits_b = MyBackboneRITS(
            n_steps=kwargs['n_steps'], 
            n_features=kwargs['n_features'], 
            rnn_hidden_size=kwargs['rnn_hidden_size']
            )

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, ...]:
        # Results from the forward RITS.
        (
            f_imputed_data,
            f_reconstruction,
            f_reconstruction_loss,
            pred_f
        ) = self.rits_f(inputs, "forward").values()
        # Results from the backward RITS.
        (
            b_imputed_data,
            b_reconstruction,
            b_reconstruction_loss,
            pred_b
        ) = self._reverse(self.rits_b(inputs, "backward").values())

        imputed_data = (f_imputed_data + b_imputed_data) / 2
        consistency_loss = self._get_consistency_loss(f_imputed_data, b_imputed_data)
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss
        
        predictions_f = torch.sigmoid(pred_f)
        predictions_b = torch.sigmoid(pred_b)
        predictions = (predictions_f + predictions_b) / 2 
        
        forward_clf_loss = F.binary_cross_entropy_with_logits(
            pred_f.view(-1), 
            inputs["label"].type_as(pred_f)
            )
        reverse_clf_loss = F.binary_cross_entropy_with_logits(
            pred_b.view(-1), 
            inputs["label"].type_as(pred_b)
            )
        clf_loss = (forward_clf_loss + reverse_clf_loss) / 2
        

        return {
            "imputed_data": imputed_data,
            "predictions": predictions,
            "f_reconstruction": f_reconstruction,
            "b_reconstruction": b_reconstruction,
            "consistency_loss": consistency_loss,
            "reconstruction_loss": reconstruction_loss,
            "clf_loss": clf_loss,
        }


    
class MultiTaskBRITS(_BRITS):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        classification_weight: float,
        reconstruction_weight: float,
    ):
        super().__init__(
            n_steps=n_steps,
            n_features=n_features,
            n_classes=n_classes,
            rnn_hidden_size=rnn_hidden_size,
            classification_weight=classification_weight,
            reconstruction_weight=reconstruction_weight,

        )
        self.model = MyBackboneBRITS(n_steps=n_steps, n_features=n_features, rnn_hidden_size=rnn_hidden_size)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        (
            imputed_data,
            predictions,
            f_reconstruction,
            b_reconstruction,
            consistency_loss,
            reconstruction_loss,
            clf_loss
        ) = self.model(inputs).values()


        loss = consistency_loss \
            + reconstruction_loss * self.reconstruction_weight \
            + clf_loss * self.classification_weight
        
        results = {
            "loss": loss,
            "imputed_data": imputed_data,
            "classification_pred": {
                "values": predictions,
                "is_sigmoid": True},
        }

        # if in training mode, return results with losses
        if training:
            results["consistency_loss"] = consistency_loss
            results["reconstruction_loss"] = reconstruction_loss
            results["classification_loss"] = clf_loss
            loss = (
                consistency_loss
                + reconstruction_loss * self.reconstruction_weight
                + clf_loss * self.classification_weight
            )

            # `loss` is always the item for backward propagating to update the model
           
            results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
            results["f_reconstruction"] = f_reconstruction
            results["b_reconstruction"] = b_reconstruction

        return results
    
