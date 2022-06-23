from fileinput import filename
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config import Config
from datetime import datetime


class TrainerFactory:
    def __init__(self, config: Config, data=None):
        self.config = config
        self.data = data
        # self.callbacks =

    def create_trainer(self, direction):
        config = self.config
        num_epochs = config.forward_num_epochs
        refresh_rate = 2
        if direction == "backward":
            num_epochs = config.backward_num_epochs
            refresh_rate = 10
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
        return [
            WandbLogger(
                name=f'{direction.title()[0]}-{self.config.model_arch}-{self.config.substrate}-{datetime.utcnow().strftime("%Y-%m-%d_%H-%M")}',
                save_dir=f"{work_folder}/wandb_logs/{direction}",
                offline=False,
                project=f"Metamaterial AI",  # {direction.title()}",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir=f"{work_folder}/test_tube_logs/{direction}",
                name=f"{direction.title()}",
            ),
        ]

    def create_callbacks(self, direction, refresh_rate):
        work_folder = self.config.work_folder
        return [
            ModelCheckpoint(
                filename='epoch={epoch:04d}-step={step}-val_loss={'+str(direction)+'/val/loss:.5f}',
                monitor=f"{direction}/val/loss",
                dirpath=f"{work_folder}/weights/{direction}",
                save_top_k=1,
                mode="min",
                save_last=True,
                auto_insert_metric_name=False
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=refresh_rate),
        ]
