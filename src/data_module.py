#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
from config import Config
from utils import FileUtils, split

LaserParams, Emiss = torch.FloatTensor, torch.FloatTensor


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: Config, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return self.create_data_loader('train')

    def val_dataloader(self) -> DataLoader:
        return self.create_data_loader('val')

    def test_dataloader(self) -> DataLoader:
        return self.create_data_loader('test')
    
    def predict_dataloader(self) -> DataLoader:
        return self.create_data_loader('predict')

    def create_data_loader(self, stage) -> DataLoader:
        shuffle = True if stage in ['train', 'predict'] else False
        return DataLoader(
            dataset=self.get_data_set(stage),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=16,
        )

    def get_data_set(self, stage):
        dataset: TensorDataset = None
        if stage == 'train':
            dataset = self.train
        elif stage in ['val', 'predict']:
            dataset = self.val
        elif stage == 'test':
            dataset = self.test
        else:
            raise Exception(f'Error: {stage} is not a valid stage.')
        return dataset


class ForwardDataModule(BaseDataModule):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__(config, config.forward_batch_size)

    def setup(self, stage: Optional[str]) -> None:
        work_folder = self.config.work_folder
        data = FileUtils.read_pt_data(
            self.config.data_folder, self.config.data_file)

        laser_params, emiss, uids = (
            data.norm_laser_params,
            data.interp_emissivities,
            data.uids,
        )
        splits = split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                laser_params[splits[s].start: splits[s].stop],
                emiss[splits[s].start: splits[s].stop],
                uids[splits[s].start: splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]
        torch.save(self.train, f"{work_folder}/forward_train_true.pt")
        torch.save(self.val, f"{work_folder}/forward_val_true.pt")
        torch.save(self.test, f"{work_folder}/forward_test_true.pt")


class BackwardDataModule(BaseDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__(config, config.backward_batch_size)

    def setup(self, stage: Optional[str]) -> None:
        work_folder = self.config.work_folder

        data = FileUtils.read_pt_data(
            self.config.data_folder, self.config.data_file)

        laser_params, emiss, uids = (
            data.norm_laser_params,
            data.interp_emissivities,
            data.uids,
        )

        splits = split(len(laser_params))

        self.train, self.val, self.test = [
            TensorDataset(
                emiss[splits[s].start: splits[s].stop],
                laser_params[splits[s].start: splits[s].stop],
                uids[splits[s].start: splits[s].stop],
            )
            for s in ("train", "val", "test")
        ]
        torch.save(self.train, f"{work_folder}/backward_train_true.pt")
        torch.save(self.val, f"{work_folder}/backward_val_true.pt")
        torch.save(self.test, f"{work_folder}/backward_test_true.pt")


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
