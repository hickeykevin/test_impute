from typing import Dict, Tuple
from torch import nn
import torch
from pypots.imputation.brits.modules.submodules import FeatureRegression
from pypots.nn.modules.rnn import TemporalDecay
from torch.autograd import Variable
from pypots.imputation.brits.modules.core import _BRITS, RITS
from pypots.imputation.usgan.modules.submodules import Discriminator 
from pypots.utils.metrics.error import calc_mae, calc_mse
import torch.nn.functional as F

class MultiTaskRITS(RITS):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.out = nn.Linear(self.rnn_hidden_size*30, self.n_classes)

        self.W_s1 = nn.Linear(kwargs['rnn_hidden_size'], 30)
        self.W_s2 = nn.Linear(30, 30)
        
    def additive_attention(self, rnn_output):
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(rnn_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def impute(self, inputs: Dict, direction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """The imputation function.
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from parameter `data`.

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
        values = inputs[direction]["X"]  # feature values
        masks = inputs[direction]["missing_mask"]  # missing masks
        deltas = inputs[direction]["deltas"]  # time-gap values

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros(
            (values.size(0), self.rnn_hidden_size), device=values.device
        )
        cell_states = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=values.device
        )

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(values.device)
        all_hidden_states = torch.zeros((values.size(0), values.size(1), self.rnn_hidden_size)).type_as(values)

        # imputation period
        for t in range(values.size(1)):
            # data shape: [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
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

            input = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                input, (hidden_states, cell_states)
            )
            all_hidden_states[:, t, :] = hidden_states

        
         # Attentions 
        attn_weight_matrix = self.additive_attention(all_hidden_states)
        hidden_matrix = torch.bmm(attn_weight_matrix, all_hidden_states)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])

        y_h = self.out(attention_output)
        label = nn.functional.one_hot(inputs['label'], self.n_classes).type_as(y_h)
        y_h_loss = nn.functional.cross_entropy(y_h, label)
        
        estimations = torch.cat(estimations, dim=1)
        imputed_data = masks * values + (1 - masks) * estimations
        return imputed_data, estimations, hidden_states, reconstruction_loss, y_h_loss, y_h
    
    def forward(self, inputs: dict, direction: str = "forward") -> dict:
        """Forward processing of the NN module.
        Parameters
        ----------
        inputs :
            The input data.

        direction :
            A keyword to extract data from parameter `data`.

        Returns
        -------
        dict,
            A dictionary includes all results.

        """
        imputed_data, estimations, hidden_state, reconstruction_loss, y_h_loss, y_h = self.impute(
            inputs, direction
        )
        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        ret_dict = {
            "consistency_loss": torch.tensor(
                0.0, device=imputed_data.device
            ),  # single direction, has no consistency loss
            "reconstruction_loss": reconstruction_loss,
            "imputed_data": imputed_data,
            "reconstructed_data": estimations,
            "final_hidden_state": hidden_state,
            "y_h_loss": y_h_loss,
            "clf_out": y_h
        }
        return ret_dict

class MultiTaskBRITS(_BRITS):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.rits_f = MultiTaskRITS(n_classes=n_classes, **kwargs)
        self.rits_b = MultiTaskRITS(n_classes=n_classes, **kwargs)

    def forward(self, inputs: Dict, training: bool = True) -> Dict:
        # Results from the forward RITS.
        ret_f = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        ret_b = self._reverse(self.rits_b(inputs, "backward"))

        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2
        reconstructed_data = (
            ret_f["reconstructed_data"] + ret_b["reconstructed_data"]
        ) / 2
        final_hidden_state = (ret_f["final_hidden_state"] + ret_b["final_hidden_state"]) / 2

        y_h_loss = ret_f['y_h_loss'] + ret_b['y_h_loss'] / 2

        results = {
            "imputed_data": imputed_data,
            "final_hidden_state": final_hidden_state,
            "y_h_loss": y_h_loss,
            "clf_out": ret_f['clf_out']
        }

        # if in training mode, return results with losses
        if training:
            consistency_loss = self._get_consistency_loss(
                ret_f["imputed_data"], ret_b["imputed_data"]
            )
            results["consistency_loss"] = consistency_loss
            loss = (
                consistency_loss
                + ret_f["reconstruction_loss"]
                + ret_b["reconstruction_loss"]
            )

            results["loss"] = loss
            results["reconstructed_data"] = reconstructed_data
            results["f_reconstructed_data"] = ret_f["reconstructed_data"]
            results["b_reconstructed_data"] = ret_b["reconstructed_data"]

        return results

    
