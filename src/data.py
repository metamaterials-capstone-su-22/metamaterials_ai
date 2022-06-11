#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib import tenumerate

import utils
from config import Config
from utils import Stage, rmse, split

LaserParams, Emiss = torch.FloatTensor, torch.FloatTensor


def get_data(
    use_cache: bool = True, num_wavelens: int = 800
) -> Tuple[LaserParams, Emiss, torch.LongTensor]:
    """Data is sorted in ascending order of wavelength."""
    if all(
        [
            use_cache,
            Path("local_data/stainless_steel.pt").exists(),
        ]
    ):
        data = torch.load(Path("local_data/stainless_steel.pt"))
        norm_laser_params, interp_emissivities, uids = (
            data["normalized_laser_params"],
            data["interpolated_emissivity"],
            data["uids"],
        )

        # XXX check length to avoid bugs.
        if interp_emissivities.shape[-1] == num_wavelens:
            return norm_laser_params, interp_emissivities, uids


class ForwardDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config.forward_batch_size

    def setup(self, stage: Optional[str]) -> None:

        laser_params, emiss, uids = get_data(
            use_cache=self.config.use_cache, num_wavelens=self.config.num_wavelens
        )
        splits = split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                laser_params[splits[s].start : splits[s].stop],
                emiss[splits[s].start : splits[s].stop],
                uids[splits[s].start : splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]
        torch.save(self.train, "local_work/forward_train_true.pt")
        torch.save(self.val, "local_work/forward_val_true.pt")
        torch.save(self.test, "local_work/forward_test_true.pt")

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


class BackwardDataModule(pl.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.batch_size = self.config.backward_batch_size

    def setup(self, stage: Optional[str]) -> None:

        # XXX we can safely use cache since ForwardDataModule is created first
        laser_params, emiss, uids = get_data(
            use_cache=True, num_wavelens=self.config.num_wavelens
        )

        splits = split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                emiss[splits[s].start : splits[s].stop],
                laser_params[splits[s].start : splits[s].stop],
                uids[splits[s].start : splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]
        torch.save(self.train, "local_work/backward_train_true.pt")
        torch.save(self.val, "local_work/backward_val_true.pt")
        torch.save(self.test, "local_work/backward_test_true.pt")

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

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )


class StepTestDataModule(pl.LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str]) -> None:
        self.test = TensorDataset(utils.step_tensor())

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test,
            batch_size=1_000,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )
