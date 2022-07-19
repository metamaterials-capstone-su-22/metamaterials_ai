from __future__ import annotations

import pytorch_lightning as pl
from pydantic import BaseModel

from config import Config
from data_module import DataModule
from models import DirectModel, InverseModel
from utils import get_latest_chk_point_path, get_specified_chk_point


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
        ckpt_path = self._get_checkpoint_path()
        self.trainer.test(
            model=self.model, ckpt_path=ckpt_path, datamodule=self.data_module
        )

    def predict(self):
        ckpt_path = self._get_checkpoint_path()
        return self.trainer.predict(
            model=self.model,
            ckpt_path=ckpt_path,
            datamodule=self.data_module,
            return_predictions=True,
        )

    def _get_checkpoint_path(self):
        chk_file_name: str = None
        # only set the file name if it is supposed to load from checkpoint
        # otherwise get the latest checkpoint
        if self.direction == "direct" and self.config.load_direct_checkpoint:
            chk_file_name = self.config.direct_saved_ckpt
        elif self.config.load_inverse_checkpoint:
            chk_file_name = self.config.inverse_saved_ckpt
        if chk_file_name:
            return get_specified_chk_point(self.config.work_folder, chk_file_name)
        else:
            return get_latest_chk_point_path(self.config.work_folder, self.direction)
