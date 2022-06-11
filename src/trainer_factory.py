import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config import Config


class TrainerFactory:
    def __init__(self, config: Config, data=None):
        self.config = config
        self.data = data

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
        work_path = self.config.work_path
        return [
            WandbLogger(
                name=f"{direction.title()} laser params",
                save_dir=f"{work_path}/wandb_logs/{direction}",
                offline=False,
                project=f"Laser {direction.title()}",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir=f"{work_path}/test_tube_logs/{direction}",
                name=f"{direction.title()}",
            ),
        ]

    def create_callbacks(self, direction, refresh_rate):
        work_path = self.config.work_path
        return [
            ModelCheckpoint(
                monitor=f"{direction}/val/loss",
                dirpath=f"{work_path}/weights/{direction}",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=refresh_rate),
        ]
