import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class Trainer():
    def __init__(self, config, direction):
        '''
            direction could be backward or forward
        '''
        self.direction = direction
        self.config = config

    def get_trainer(self):
        return pl.Trainer(
            max_epochs=self.get_num_of_epochs(),
            logger=self.get_loggers(),
            callbacks=self.get_callbacks(),
            gpus=1,
            precision=32,
            weights_summary="full",
            check_val_every_n_epoch=self.get_check_val_interval(),
            gradient_clip_val=0.5,
            log_every_n_steps=self.get_log_interval(),
        )

    def get_num_of_epochs(self):
        return self.config.forward_num_epochs \
            if self.direction == 'forward' \
            else self.config.backward_num_epochs

    def get_check_val_interval(self):
        num_epochs = self.get_num_of_epochs()
        return min(3, num_epochs - 1)

    def get_log_interval(self):
        ''' The same as check_val interval'''
        return self.get_check_val_interval()

    def get_loggers(self):
        direction = self.direction
        work_path = self.config.work_path
        title = direction.title()
        return [
            WandbLogger(
                name=f"{title} laser params",
                save_dir=f"{work_path}/wandb_logs/{direction}",
                offline=False,
                project=f"Laser {title}",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir=f"{work_path}/test_tube_logs/{direction}",
                name=f"{title}",
            ),
        ]

    def get_callbacks(self):
        direction = self.direction
        work_path = self.config.work_path

        return [
            ModelCheckpoint(
                monitor=f"{direction}/val/loss",
                dirpath=f"{work_path}/weights/{direction}",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            pl.callbacks.progress.TQDMProgressBar(refresh_rate=2),
        ]
