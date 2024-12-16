from lightning.pytorch import Callback
from torchmetrics.classification import F1Score
from torchmetrics import MetricCollection
from typing import Dict
import torch
from torch.nn import functional as F
from torch import nn
from src.utils import get_pylogger

log = get_pylogger("cv_metrics.py")

class ScoreCrossValidationCallback:
    def __init__(self, num_classes: int, average: str = 'macro'):
        super().__init__()
        self.metrics = MetricCollection({
            'f1': F1Score(
                task='binary' if num_classes == 2 else 'multiclass',
                num_classes=num_classes, 
                average=average
                )
        })

        self.fold_scores = {
            k: [] for k in self.metrics.keys()
        }

    def on_train_batch_start(self, *args, **kwargs):
        pass

    def convert_logits(self, logits):
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        if torch.allclose(logits.sum(dim=1), torch.ones(logits.shape[0], device=logits.device), atol=1e-2):
            return torch.argmax(logits, dim=1)
        else:
            return torch.argmax(F.softmax(logits, dim=1), dim=1)

    def convert_target(self, target):
        pass

    def on_validation_epoch_start(self, *args, **kwargs):
        self.metrics.reset()

    def on_validation_batch_end(self, batch, outputs, **kwargs):
        model_out: Dict = outputs['clf_logits']
        # assert there is a key in predictions called is_sigmoid
        assert 'is_sigmoid' in model_out.keys()

        if model_out['is_sigmoid']:
            predictions = model_out['values']
        else:
            predictions = self.convert_logits(model_out["values"])
        
        target = batch["label"]
        self.metrics.update(predictions.view(-1), target.view(-1))

    def on_validation_epoch_end(self, *args, **kwargs):
        metrics: Dict = self.metrics.compute()
        for k, metric in metrics.items():
            self.fold_scores[k].append(metric)
        self.metrics.reset()

    def on_fit_end(self, *args, **kwargs):
        self.metrics.reset()
        log.info(f"fold scores: {self.fold_scores}")
        self.fold_scores = {
            k: torch.stack(v).mean() for k, v in self.fold_scores.items()
        }