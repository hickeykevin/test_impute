from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from src.utils import MeanRelativeError
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryF1Score
from torchmetrics import MetricCollection
from torchmetrics.wrappers import BootStrapper
from torch.nn import functional as F
import torch


class ClassificationMetricsCallback(Callback):
    def __init__(self, boot_val: bool):
        self.boot_val = boot_val
    
    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self.train_clf_metrics = MetricCollection({
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "auc": BinaryAUROC(),
                "f1": BinaryF1Score()
            }).to(pl_module.device)
            self.val_clf_metrics = MetricCollection({
                "precision": BinaryPrecision() if not self.boot_val else BootStrapper(BinaryPrecision(), num_bootstraps=1000, sampling_strategy="multinomial"),
                "recall": BinaryRecall() if not self.boot_val else BootStrapper(BinaryRecall(), num_bootstraps=1000, sampling_strategy="multinomial"),
                "auc":  BinaryAUROC() if not self.boot_val else BootStrapper(BinaryAUROC(), num_bootstraps=1000, sampling_strategy="multinomial"),
                "f1": BinaryF1Score() if not self.boot_val else BootStrapper(BinaryF1Score(), num_bootstraps=1000, sampling_strategy="multinomial")
            }).to(pl_module.device) 
        if stage == "test":
            self.test_clf_metrics = MetricCollection({
                "precision": BootStrapper(BinaryPrecision(), num_bootstraps=1000, sampling_strategy="multinomial"),
                "recall": BootStrapper(BinaryRecall(), num_bootstraps=1000, sampling_strategy="multinomial"),
                "auc": BootStrapper(BinaryAUROC(), num_bootstraps=1000, sampling_strategy="multinomial"),
                "f1": BootStrapper(BinaryF1Score(), num_bootstraps=1000, sampling_strategy="multinomial")
            }).to(pl_module.device)

        self.no_val = trainer.datamodule.hparams.val_ratio == 0.0 and not trainer.datamodule.hparams.ricardo

    
    def convert_logits(self, logits):
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        # if every row of logits adds to 1, then it already is softmaxed
        if torch.allclose(logits.sum(dim=1), torch.ones(logits.shape[0], device=logits.device), atol=1e-2):
            return torch.argmax(logits, dim=1) #MultiTaskBRITS
        else:
            return torch.argmax(F.softmax(logits, dim=1), dim=1) 
    
    def convert_target(self, target):
        return target.view(-1, 1)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        clf_logits = self.convert_logits(outputs["clf_logits"])
        target = self.convert_target(batch["label"])
        self.train_clf_metrics.update(clf_logits.view(-1, 1), target)
        
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = self.train_clf_metrics.compute()
        for k, v in metrics.items():
            pl_module.log(f"train/{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_clf_metrics.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.no_val:
            return
        else:
            clf_logits = self.convert_logits(outputs["clf_logits"])
            target = self.convert_target(batch["label"])
            self.val_clf_metrics.update(clf_logits.view(-1, 1), target)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.no_val:
            return
        else:
            metrics = self.val_clf_metrics.compute()
            for k, v in metrics.items():
                if "_mean" in k:
                    k = k.replace("_mean", "")
                pl_module.log(f"val/{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)
            self.val_clf_metrics.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        clf_logits = self.convert_logits(outputs["clf_logits"])
        target = self.convert_target(batch["label"])
        self.test_clf_metrics.update(clf_logits, target)

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = self.test_clf_metrics.compute()
        for k, v in metrics.items():
            pl_module.log(f"test/{k}", v, on_epoch=True, prog_bar=True)
        self.test_clf_metrics.reset()

class CVClassificationMetricsCallback(ClassificationMetricsCallback):
    def __init__(self, boot_val: bool):
        super().__init__(boot_val)
    
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = self.train_clf_metrics.compute()
        return metrics
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.no_val:
            return
        else:
            metrics = self.val_clf_metrics.compute()
            return metrics
        

       
class ImputationMetricsCallback(Callback):
    def __init__(self, boot_val: bool):
        self.boot_val = boot_val
    
    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self.train_mre = MeanRelativeError().to(pl_module.device) 
            self.val_mre = MeanRelativeError().to(pl_module.device) if not self.boot_val else BootStrapper(MeanRelativeError(), num_bootstraps=1000, sampling_strategy="multinomial").to(pl_module.device)
        elif stage == "test":
            self.test_mre = BootStrapper(MeanRelativeError(), num_bootstraps=1000, sampling_strategy="multinomial").to(pl_module.device)

        self.no_val = trainer.datamodule.hparams.val_ratio == 0.0 and not trainer.datamodule.hparams.ricardo
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        imputed_data = outputs["imputed_data"]
        target = outputs["X_ori"]
        mask = outputs["indicating_mask"]
        self.train_mre(imputed_data, target, mask)
        
    def on_train_epoch_end(self, trainer, pl_module):
        mre = self.train_mre.compute()
        pl_module.log("train/mre", mre, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_mre.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.no_val:
            pass
        else:
            imputed_data = outputs["imputed_data"]
            target = outputs["X_ori"]
            mask = outputs["indicating_mask"]
            self.val_mre(imputed_data, target, mask)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.no_val:
           pass
        else:
            mre = self.val_mre.compute()
            if not self.boot_val:
                pl_module.log("val/mre", mre, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                pl_module.log("val/mre", mre['mean'], on_epoch=True, prog_bar=True, sync_dist=True)
                pl_module.log("val/mre_std", mre['std'], on_epoch=True, prog_bar=True, sync_dist=True)
            self.val_mre.reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        imputed_data = outputs["imputed_data"]
        target = outputs["X_ori"]
        mask = outputs["indicating_mask"]
        self.test_mre(imputed_data, target, mask)

    def on_test_epoch_end(self, trainer, pl_module):
        mre = self.test_mre.compute()
        pl_module.log("test/mre", mre['mean'], on_epoch=True, prog_bar=True)
        pl_module.log("test/mre_ci", mre['std'], on_epoch=True, prog_bar=True)
        self.test_mre.reset()