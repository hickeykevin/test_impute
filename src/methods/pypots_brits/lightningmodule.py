from lightning.pytorch import LightningModule
from pypots.classification.brits.core import _BRITS

class PyPotsBRITS(LightningModule):
    def __init__(
            self, 
            rnn_hidden_size,
            classification_weight,
            reconstruction_weight,
            lr,
            **kwargs
    )
        super().__init__(**kwargs)
        self.model = _BRITS(
            rnn_hidden_size=rnn_hidden_size,
            classification_weight=classification_weight,
            reconstruction_weight=reconstruction_weight,
            lr=lr
        )
