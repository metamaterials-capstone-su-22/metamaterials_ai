
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import DataUtils
from config import Config


class ForwardDataModule(pl.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config.forward_batch_size

    def setup(self, stage: Optional[str]) -> None:
        data_path= self.config.data_path
        laser_params, emiss, uids = DataUtils.get_data(
            use_cache=self.config.use_cache, num_wavelens=self.config.num_wavelens
        )
        splits = DataUtils.split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                laser_params[splits[s].start: splits[s].stop],
                emiss[splits[s].start: splits[s].stop],
                uids[splits[s].start: splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]
        torch.save(self.train, f"{data_path}/forward_train_true.pt")
        torch.save(self.val, f"{data_path}/forward_val_true.pt")
        torch.save(self.test, f"{data_path}/forward_test_true.pt")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )
