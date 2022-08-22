from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config import Config


class TrainerFactory:
    def __init__(self, config: Config, data=None):
        self.config = config
        self.data = data

    def create_trainer(self, direction):
        config = self.config
        if direction == "inverse":
            num_epochs = config.inverse_num_epochs
            refresh_rate = 2
            self.model_arch = config.inverse_arch
        else:
            num_epochs = config.direct_num_epochs
            refresh_rate = 2
            self.model_arch = config.direct_arch
        return self.create_pl_trainer(direction, refresh_rate, num_epochs)

    def create_pl_trainer(self, direction, refresh_rate, epochs):
        return pl.Trainer(
            max_epochs=epochs,
            logger=self.create_loggers(direction),
            callbacks=self.create_callbacks(direction, refresh_rate),
            gpus=1,
            precision=32,
            weights_summary="full",
            check_val_every_n_epoch=min(3, epochs - 1),
            gradient_clip_val=0.5,
            log_every_n_steps=min(3, epochs - 1),
        )

    def create_loggers(self, direction):
        work_folder = self.config.work_folder
        loggers = [
            TensorBoardLogger(
                save_dir=f"{work_folder}/test_tube_logs/{direction}",
                name=f"{direction.title()}",
            )
        ]
        # Do not log the loaded model
        if not self.is_loading_from_checkpoint(direction):
            loggers.append(
                WandbLogger(
                    name=f'{direction.title()[0]}-{self.config.data_portion}-{self.model_arch}-{self.config.substrate}-{datetime.utcnow().strftime("%Y-%m-%d_%H-%M")}',
                    save_dir=f"{work_folder}/wandb_logs/{direction}",
                    offline=False,
                    project=f"Metamaterial AI",
                    log_model=True,
                )
            )
        return loggers

    def is_loading_from_checkpoint(self, direction):
        return (direction == "direct" and self.config.load_direct_checkpoint) or (
            direction == "inverse" and self.config.load_inverse_checkpoint
        )

    def create_callbacks(self, direction, refresh_rate):
        callbacks = [
            self.create_checkpoint_callback(direction),
            LearningRateMonitor(logging_interval="epoch"),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=refresh_rate),
        ]
        if self.config.enable_early_stopper:
            callbacks.append(TrainerFactory.create_early_stopper_callback(direction))

        return callbacks

    def create_checkpoint_callback(self, direction):
        work_folder = self.config.work_folder
        return ModelCheckpoint(
            filename="epoch={epoch:04d}-step={step}-val_loss={"
            + str(direction)
            + "/val/loss:.5f}",
            monitor=f"{direction}/val/loss",
            dirpath=f"{work_folder}/weights/{direction}",
            save_top_k=1,
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        )

    @staticmethod
    def create_early_stopper_callback(direction):
        return EarlyStopping(
            monitor=f"{direction}/val/loss",
            strict=True,
            check_on_train_epoch_end=False,
            patience=10,
            min_delta=0.000_1,
            verbose=True,
            mode="min",
        )
