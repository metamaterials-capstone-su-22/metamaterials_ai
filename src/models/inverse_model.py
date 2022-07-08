from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from utils import rmse

from .base_model import BaseModel
from .direct_model import DirectModel
from .model_arch_factory import ModelArchFactory
from .model_config import ModelConfig


class InverseModel(BaseModel):
    def __init__(
        self,
        config: Config,
        direct_model: Optional[DirectModel] = None,
    ):
        config = Config.parse_obj(config) if type(config) is dict else config

        self.model_config = ModelConfig(
            arch=config.inverse_arch,
            direction="inverse",
            num_classes=14,
            in_channels=config.num_wavelens,
        )
        super().__init__(config, direction="inverse")
        self.lr = config.inverse_lr
        self.milestones = [25, 50, 130, 140, 150]
        self.example_input_array = torch.randn(1, 800)
        self.save_hyperparameters(config.__dict__)

        if direct_model is None:
            self.direct_model = None
        else:
            self.direct_model = direct_model
            self.direct_model.freeze()

    def create_model_arc(self):
        return ModelArchFactory.create_model_arch(self.model_config)

    def initialize_model(self):
        self.continuous_head = nn.LazyLinear(2)
        self.discrete_head = nn.LazyLinear(12)
        # This call *must* happen to initialize the lazy layers
        _dummy_input = torch.rand(2, self.config.num_wavelens)
        self.forward(_dummy_input)

    def forward(self, x):
        h = self.model(x)
        laser_params = torch.sigmoid(self.continuous_head(h))
        wattages = F.one_hot(
            torch.argmax(self.discrete_head(h), dim=-1), num_classes=12
        )

        return torch.cat((laser_params, wattages), dim=-1)

    def predict_step(self, batch, _batch_nb):
        out = {"params": None, "pred_emiss": None, "pred_loss": None}
        # If step data, there's no corresponding laser params
        try:
            (y,) = batch  # y is emiss
        except ValueError:
            (y, x, uids) = batch  # y is emiss,x is laser_params
            out["true_params"] = x
            out["uids"] = uids
        out["true_emiss"] = y
        x_pred = self(y)
        out["params"] = x_pred
        if self.direct_model is not None:
            y_pred = self.direct_model(x_pred)
            out["pred_emiss"] = y_pred
            y_loss = rmse(y_pred, y)
            out["pred_loss"] = y_loss
            loss = y_loss
        return out

    def training_step(self, batch, _batch_nb):
        return self.execute_step(batch, "train")

    def validation_step(self, batch, batch_nb):
        return self.execute_step(batch, "val")

    def test_step(self, batch, batch_nb):
        return self.execute_step(batch, "test")

    def execute_step(self, batch, stage):
        y, x, uids = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = rmse(x_pred, x)
            self.log(f"{self.direction}/{stage}/x/loss", x_loss, prog_bar=True)
        if self.direct_model is not None:
            y_pred = self.direct_model(x_pred)
            y_loss = rmse(y_pred, y)

            self.log(
                f"{self.direction}/{stage}/y/loss",
                y_loss,
                prog_bar=True,
            )

            loss = y_loss
            if stage == "test":
                self.save_test_result(x, y, y_pred, x_pred)
        # else: #TODO
        # add logic to snag the current best forward model
        # if the case that we are exclusively training the inverse model

        self.create_graph_and_log(stage, y_pred, y, loss)
        return loss

    def save_test_result(self, x, y, y_pred, x_pred):
        torch.save(x, f"{self.work_folder}/params_true_back.pt")
        torch.save(y, f"{self.work_folder}/emiss_true_back.pt")
        torch.save(y_pred, f"{self.work_folder}/emiss_pred.pt")
        torch.save(x_pred, f"{self.work_folder}/param_pred.pt")
