import os
from typing import Union
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def get_tensorboard_logger(loggers: List[Logger]):
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger


class GenerationCallback(Callback):
    """
    Callback for generating imputations and saving plots during validation.

    Args:
        log_dir (str): The directory path where the callback outputs will be saved.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir) / "callback_outputs"

    @rank_zero_only
    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.tb_logger: TensorBoardLogger = get_tensorboard_logger(trainer.loggers)
    
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Callback function called at the end of each validation batch during training.

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module being trained.
            outputs: The outputs from the model for the current batch, from `validation_step`.
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        if self.tb_logger and trainer.is_last_batch:
            imputation: np.ndarray = outputs["imputed_data"][0].detach().cpu().numpy()
            eval_target: np.ndarray = outputs["X_ori"][0].detach().cpu().numpy()
            eval_mask: np.ndarray = outputs["indicating_mask"][0].detach().cpu().numpy()
            if hasattr(trainer.datamodule.hparams, 'normalize') and trainer.datamodule.hparams.normalize: #TODO fix this so that it applies to both physionet and daicwoz
                imputation = trainer.datamodule.inverse_normalize_data(np.expand_dims(imputation, 0), as_tensor=False)
                imputation = np.squeeze(imputation)
                eval_target = trainer.datamodule.inverse_normalize_data(np.expand_dims(eval_target, 0), as_tensor=False)
                eval_target = np.squeeze(eval_target)
            for i in range(10 if imputation.shape[-1] > 10 else imputation.shape[-1]):
                fig = plt.figure(figsize=(10, 8))
                # Plot the full imputations and eval targets, 
                # generations for missing_indices marked in red, and eval targets marked in green
                plt.plot(range(imputation.shape[0]), imputation[:, i], color='blue')                
                plt.plot(range(eval_target.shape[0]), eval_target[:, i], color='orange')
                for ts in range(imputation.shape[0]):
                    if eval_mask[ts, i] == 1:
                        plt.scatter(ts, imputation[ts, i], color='red', marker='o')
                        plt.scatter(ts, eval_target[ts, i], color='green', marker='o')

                self.tb_logger.experiment.add_figure(f"imputation_{i}", plt.gcf(), global_step=trainer.global_step)
                plt.close()