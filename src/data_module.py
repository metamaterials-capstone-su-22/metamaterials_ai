from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
from config import Config
from dto.data import Data
from utils import FileUtils, split
from utils.utils import Stage


class DataModule(pl.LightningDataModule):
    def __init__(self, config: Config, direction: str) -> None:
        super().__init__()
        self.config = config
        self.direction = direction
        self.batch_size = (
            config.direct_batch_size
            if direction == "direct"
            else config.inverse_batch_size
        )

    def setup(self, stage: Optional[str]) -> None:
        data: Data = FileUtils.read_pt_data(
            self.config.data_folder, self.config.data_file
        )
        splits = split(len(data.laser_params) * self.config.data_portion)
        if self.direction == "direct":
            self.direct_split(data, splits)
        else:
            self.inverse_split(data, splits)
        self.save_split_data()

    def train_dataloader(self) -> DataLoader:
        return self.create_data_loader("train")

    def val_dataloader(self) -> DataLoader:
        return self.create_data_loader("val")

    def test_dataloader(self) -> DataLoader:
        return self.create_data_loader("test")

    def predict_dataloader(self) -> DataLoader:
        return self.create_data_loader("predict")

    def create_data_loader(self, stage) -> DataLoader:
        shuffle = True if stage in ["train", "predict"] else False
        return DataLoader(
            dataset=self.get_data_set(stage),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=16,
        )

    def get_data_set(self, stage):
        dataset: TensorDataset = None
        if stage == "train":
            dataset = self.train
        elif stage in ["val", "predict"]:
            dataset = self.val
        elif stage == "test":
            dataset = self.test
        else:
            raise Exception(f"Error: {stage} is not a valid stage.")
        return dataset

    def save_split_data(self):
        work_folder = self.config.work_folder
        direction = self.direction
        torch.save(self.train, f"{work_folder}/{direction}_train_true.pt")
        torch.save(self.val, f"{work_folder}/{direction}_val_true.pt")
        torch.save(self.test, f"{work_folder}/{direction}_test_true.pt")

    def inverse_split(self, data: Data, splits: dict[Stage, range]):
        self.train, self.val, self.test = [
            TensorDataset(
                data.emiss[splits[s].start: splits[s].stop],
                data.laser_params[splits[s].start: splits[s].stop],
                data.uids[splits[s].start: splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]

    def direct_split(self, data: Data, splits: dict[Stage, range]):
        self.train, self.val, self.test = [
            TensorDataset(
                data.laser_params[splits[s].start: splits[s].stop],
                data.emiss[splits[s].start: splits[s].stop],
                data.uids[splits[s].start: splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]


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
