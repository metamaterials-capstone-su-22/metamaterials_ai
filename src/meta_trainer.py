from __future__ import annotations

import pytorch_lightning as pl
from pydantic import BaseModel

from config import Config
from data_module import DataModule
from models import InverseModel, DirectModel
from utils import get_latest_chk_point_path


class MetaTrainer(BaseModel):
    """This class wraps the model, trainer, and datamodule
    for each direction (direct and inverse)"""

    config: Config
    model: DirectModel | InverseModel
    data_module: DataModule
    trainer: pl.Trainer
    direction: str

    class Config:
        arbitrary_types_allowed = True

    def fit(self):
        self.trainer.fit(model=self.model, datamodule=self.data_module)

    def test(self):
        ckpt_path = get_latest_chk_point_path(
            self.config.work_folder, self.direction)
        self.trainer.test(
            model=self.model, ckpt_path=ckpt_path, datamodule=self.data_module
        )

    def predict(self):
        ckpt_path = get_latest_chk_point_path(
            self.config.work_folder, self.direction)
        return self.trainer.predict(
            model=self.model,
            ckpt_path=ckpt_path,
            datamodule=self.data_module,
            return_predictions=True,
        )
