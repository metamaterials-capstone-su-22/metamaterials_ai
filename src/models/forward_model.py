import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from utils import TrainingUtils

from .model_config import ModelConfig
from .model_factory import ModelFactory

rmse = TrainingUtils.rmse


class Model(pl.LightningModule):
    def __init__(self, config: Config, model_config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = ModelFactory.get_model(model_config)
        self.config = config
        self.model_config = model_config
        self.initialize()
        self.step_func = (
            self.forward_step
            if self.model_config.direction == "forward"
            else self.backward_step
        )

    def initialize(self):
        _dummy_input = None
        if self.model_config.direction == "forward":
            _dummy_input = torch.rand(2, 14)
        else:
            self.continuous_head = nn.LazyLinear(2)
            self.discrete_head = nn.LazyLinear(12)
            # call *must* happen to initialize the lazy layers
            _dummy_input = torch.rand(2, self.config.num_wavelens)
        self.forward(_dummy_input)

    def forward(self, x):
        return (
            self.model(x)
            if self.model_config.direction == "forward"
            else self.forward_of_backward_model(x)
        )

    def predict_step(self, batch, _batch_nb):
        if self.model_config.direction == "forward":
            raise Exception("forward direction predict is not implemented!")
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
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            out["pred_emiss"] = y_pred
            y_loss = rmse(y_pred, y)
            out["pred_loss"] = y_loss
        return out

    def forward_of_backward_model(self, x):
        h = self.model(x)
        laser_params = torch.sigmoid(self.continuous_head(h))
        wattages = F.one_hot(
            torch.argmax(self.discrete_head(h), dim=-1), num_classes=12
        )
        return torch.cat((laser_params, wattages), dim=-1)

    def backward_step(self, batch, batch_nb, stage: str):
        y, x, uids = batch

        x_pred = self(y)
        with torch.no_grad():
            x_loss = rmse(x_pred, x)
            self.log(f"backward/{stage}/x/loss", x_loss, prog_bar=True)
        if self.forward_model is not None:
            y_pred = self.forward_model(x_pred)
            y_loss = rmse(y_pred, y)

            self.log(
                f"backward/{stage}/y/loss",
                y_loss,
                prog_bar=True,
            )

            loss = y_loss
        self.log_and_graph(y_pred, y, loss, stage)
        return loss

    def forward_step(self, batch, batch_nb, stage: str):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log_and_graph(y_pred, y, loss, stage)
        return loss

    def training_step(self, batch, batch_nb):
        return self.step_func(batch, batch_nb, "train")

    def validation_step(self, batch, batch_nb):
        return self.step_func(batch, batch_nb, "val")

    def test_step(self, batch, batch_nb):
        return self.step_func(batch, batch_nb, "test")

    def configure_optimizers(self):
        lr = (
            self.config.forward_lr
            if self.model_config.direction == "forward"
            else self.config.backward_lr
        )
        return torch.optim.AdamW(self.parameters(), lr=lr)

    def log_and_graph(self, y_pred, y, loss, stage):
        direction = self.model_config.direction
        self.log(f"{direction}/{stage}/loss", loss, prog_bar=True)
        # TODO: question: why last 5 ?
        # if self.current_epoch > self.config["forward_num_epochs"] - 5:
        #     nngraph.save_integral_emiss_point(
        #         y_pred, y, "/data-new/spencersong/forwards_val_points.txt", all_points=True
        #     )
