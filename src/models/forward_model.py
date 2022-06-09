from torch import  nn
from .mixer_module import MLPMixer
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from config import Config
import pytorch_lightning as pl
from utils import TrainingUtils

rmse = TrainingUtils.rmse

class ForwardModel(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # self.config["num_wavelens"] = len(
        #     torch.load(Path("/data-new/spencersong/data.pt"))["interpolated_wavelength"][0]
        # )
        # self.save_hyperparameters()
        self.model = nn.Sequential(
            Rearrange("b c -> b c 1 1"),
            MLPMixer(
                in_channels=14,
                image_size=1,
                patch_size=1,
                num_classes=self.config.num_wavelens,
                dim=512,
                depth=8,
                token_dim=256,
                channel_dim=2048,
                dropout=0.5,
            ),
            nn.Sigmoid(),
        )

        # TODO how to reverse the *data* in the Linear layers easily? transpose?
        # XXX This call *must* happen to initialize the lazy layers
        self.forward(torch.rand(2, 14))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log_and_graph(y_pred, y, loss, "train")
        return loss

    def validation_step(self, batch, batch_nb):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log_and_graph(y_pred, y, loss, "val")
        return loss

    def test_step(self, batch, batch_nb):
        x, y, uids = batch
        y_pred = self(x)
        loss = rmse(y_pred, y)
        self.log_and_graph(y_pred, y, loss, "test")
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.forward_lr)

    def log_and_graph(self, y_pred, y, loss, stage):
        self.log(f"forward/{stage}/loss", loss, prog_bar=True)
        # TODO: question: why last 5 ?
        # if self.current_epoch > self.config["forward_num_epochs"] - 5:
        #     nngraph.save_integral_emiss_point(
        #         y_pred, y, "/data-new/spencersong/forwards_val_points.txt", all_points=True
        #     )
        return
