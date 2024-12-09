from pypots.nn.modules.csdi.backbone import BackboneCSDI
from pypots.nn.modules.csdi.layers import CsdiDiffusionModel, CsdiResidualBlock
import torch
import math
import torch.nn.functional as F
from torch import nn

class MultiTaskCsdiResidualBlock(CsdiResidualBlock):
    def __init__(self, d_side, n_channels, diffusion_embedding_dim, nheads):
        super().__init__(d_side, n_channels, diffusion_embedding_dim, nheads)
    
    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)

        y = x + diffusion_emb
        y = self.forward_time(y, base_shape)
        feature_out = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(feature_out)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip, feature_out
    
class MultiTaskDiffusionModel(CsdiDiffusionModel):
    def __init__(
        self,
        n_diffusion_steps,
        d_diffusion_embedding,
        d_input,
        d_side,
        n_channels,
        n_heads,
        n_layers,
    ):
        super().__init__(n_diffusion_steps, d_diffusion_embedding, d_input, d_side, n_channels, n_heads, n_layers)

        self.residual_layers = nn.ModuleList(
            [
                MultiTaskCsdiResidualBlock(
                    d_side=d_side,
                    n_channels=n_channels,
                    diffusion_embedding_dim=d_diffusion_embedding,
                    nheads=n_heads,
                )
                for _ in range(n_layers)
            ]
        )
    def forward(self, x, cond_info, diffusion_step):
        (
            n_samples,
            input_dim,
            n_features,
            n_steps,
        ) = x.shape  # n_samples, 2, n_features, n_steps

        x = x.reshape(n_samples, input_dim, n_features * n_steps)
        x = self.input_projection(x)  # n_samples, n_channels, n_features*n_steps
        x = F.relu(x)
        x = x.reshape(n_samples, self.n_channels, n_features, n_steps)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        feature_out = []
        for layer in self.residual_layers:
            x, skip_connection, feature_layer_out = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
            feature_out.append(feature_layer_out)
        feature_out = torch.cat(feature_out[::len(self.residual_layers)], dim=0)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(n_samples, self.n_channels, n_features * n_steps)
        x = self.output_projection1(x)  # (n_samples, channel, n_features*n_steps)
        x = F.relu(x)
        x = self.output_projection2(x)  # (n_samples, 1, n_features*n_steps)
        x = x.reshape(n_samples, n_features, n_steps)
        return x, feature_out
    
class MultiTaskBackboneCSDI(BackboneCSDI):
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        # clf components
        self.W_s1 = nn.Linear(kwargs['d_feature_embedding'], 30) #(B,channel,K*L)
        self.W_s2 = nn.Linear(30, 30)
        self.out = nn.Linear(kwargs['d_feature_embedding']*30, 2)
        

        d_side = kwargs['d_time_embedding'] + kwargs['d_feature_embedding']
        if kwargs['is_unconditional']:
            d_input = 1
        else:
            d_side += 1  # for conditional mask
            d_input = 2
                
        self.diff_model = MultiTaskDiffusionModel(
            n_diffusion_steps = kwargs['n_diffusion_steps'],
            d_diffusion_embedding = kwargs['d_diffusion_embedding'],
            d_input = d_input,
            d_side = d_side,
            n_channels = kwargs['n_channels'],
            n_heads = kwargs['n_heads'],
            n_layers = kwargs['n_layers'],
        )

    def additive_attention(self, rnn_output):
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(rnn_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    # def get_embedding(self, diffusion_step):
    #     return self.diff_model.diffusion_embedding(diffusion_step)

    def calc_loss_valid(self, observed_data, target, cond_mask, indicating_mask, side_info, is_train):

        loss_sum = 0
        clf_loss = 0
        clf_output = []
        for t in range(self.n_diffusion_steps):
            loss, clf_loss, residual, clf_out = self.calc_loss(
                observed_data, target, cond_mask, indicating_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
            clf_loss += clf_loss
            if t == self.n_diffusion_steps - 1:
                clf_output.append(clf_out)
        clf_output = torch.cat(clf_output, dim=0)
        return (loss_sum / self.n_diffusion_steps), clf_loss, residual, clf_output

    def calc_loss(self, observed_data, target, cond_mask, indicating_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        device = observed_data.device
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(device)
        else:
            t = torch.randint(0, self.n_diffusion_steps, [B]).to(device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted, feature_out = self.diff_model(total_input.float(), side_info.float(), t)  # (B,K,L)
        attn_weight_matrix = self.additive_attention(feature_out.permute(0, 2, 1))
        hidden_matrix = torch.bmm(attn_weight_matrix, feature_out.permute(0, 2, 1))
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])

        clf_out = self.out(attention_output)
        clf_loss = F.cross_entropy(clf_out, target.long())
        
        target_mask = indicating_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss, clf_loss, residual, clf_out
    



        