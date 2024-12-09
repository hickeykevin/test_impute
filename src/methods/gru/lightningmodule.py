import torch 
from torch import nn 
from lightning.pytorch import LightningModule
import torch.nn.functional as F
from typing import Optional, List, Dict, Union


class GRU_base(nn.Module):
    def __init__(
            self, 
            n_features: int,
            hidden_dim: int,
            n_layers: int, 
            dropout_prob: float,
            n_classes: int,
            use_biGRU: bool, 
        ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.n_classes = n_classes
        self.use_biGRU = use_biGRU
        
        D = 2 if use_biGRU else 1   # for Bidirectional case

        # --- Mappings ---
        self.RNNCell = nn.GRU(n_features, hidden_dim, n_layers, dropout=dropout_prob,
                             bidirectional=use_biGRU, batch_first=True)
        self.predict = nn.Linear(D * hidden_dim, n_classes)

        #activation function
        self.act = nn.Sigmoid()

    def initHidden(self, bsz):
        """Initialize hidden states"""
        D = 2 if self.use_biGRU else 1   # for Bidirectional case
        h = torch.zeros(D * self.n_layers,
                        bsz,
                        self.hidden_dim,
                        requires_grad=True)
        return h
    
    def forward(self, X):
        """
        The main method of your model, completely mapping the input data to the
        output predictions. In this example, the RNN outputs a classification
        using only the final hidden state (out[-1]).
        """
        B, T, V = X.shape # Assume timesteps x batch x variables input
        #print('self._N_FEATURES:', self._N_FEATURES)

        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(X, hidden)

        y_hat = self.predict(out[:, -1, :])

        # y_hat = self.act(y_hat)  # between 0 and 1

        return y_hat
    


class GRULightningModule(LightningModule):
    def __init__(self, hidden_dim, n_layers, dropout_prob, use_biGRU):
        super().__init__()
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()

    def setup(self, stage: Optional[str] = "fit"):
        data_info = self.trainer.datamodule.data_info
        assert bool([x in data_info.keys() for x in ['n_features', 'n_classes']]), "data_info should contain 'n_features' and 'n_classes"
        n_features = data_info["n_features"]
        n_classes = data_info["n_classes"]

        self.model = GRU_base(
            n_features=n_features,
            hidden_dim=self.hparams.hidden_dim, 
            n_layers=self.hparams.n_layers, 
            dropout_prob=self.hparams.dropout_prob, 
            n_classes=n_classes, 
            use_biGRU=self.hparams.use_biGRU
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = self.assemple_input(batch)
        y_hat = self(x)
        y = y.long()
        loss = self.loss(y_hat, y)
        self.log('train/loss', loss)
        return {
            'loss': loss,
            'clf_logits': y_hat,
            'X_ori': x,
        }

    def validation_step(self, batch, batch_idx):
        x, y = self.assemple_input(batch)
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val/loss', loss)
        return {
            'loss': loss,
            'clf_logits': y_hat,
            'X_ori': x,
        }

    def test_step(self, batch, batch_idx):
        x, y = self.assemple_input(batch)
        y = y.long()
        y_hat = self(x)
        return {
            'clf_logits': y_hat,
            'X_ori': x,
        }
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def assemple_input(self, data):
        # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
        X = data['X_ori'].type_as(next(iter(self.model.parameters())))
        label = data['label']
        return X, label

