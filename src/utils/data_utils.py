from typing import Dict, List, Literal, Mapping, Optional, TypedDict
from math import floor
from config import Config
from pathlib import Path
import torch
from data_modules import Data


class DataUtils():
    Stage = Literal["train", "val", "test"]
    # TODO (Based on old code note) replace with scanl

    @staticmethod
    def split(n: int, splits: Optional[Mapping[Stage, float]] = None) -> Dict[Stage, range]:
        """
        n: length of dataset
        splits: map where values should sum to 1 like in `{"train": 0.8, "val": 0.1, "test": 0.1}`
        """
        if splits is None:
            splits = {"train": 0.8, "val": 0.1, "test": 0.1}
        return {
            "train": range(0, floor(n * splits["train"])),
            "val": range(
                floor(n * splits["train"]),
                floor(n * splits["train"]) + floor(n * splits["val"]),
            ),
            "test": range(floor(n * splits["train"]) + floor(n * splits["val"]), n),
        }

    @staticmethod
    def read_pt_data(config: Config):
        '''Read data from pt file (saved as pytorch tensor)'''
        data_file = Path(f'{config.data_path}/{config.substrate}.pt')
        try:
            data = torch.load(data_file)
        except Exception as e:
            print(
                f'Trouble in loading data file: {data_file}. Error: {e.message}')

        # Set the number of wavelengths
        config.num_wavelens = data["interpolated_emissivity"].shape[-1]
        return Data(norm_laser_params=data["normalized_laser_params"], interp_emissivities=data["interpolated_emissivity"], uids=data["uids"], wavelength=data["wavelength"])
