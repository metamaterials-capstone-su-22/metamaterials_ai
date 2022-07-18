import shutil
from getpass import getpass
from pathlib import Path

import gdown
import torch

from dto import Data

from .utils import get_dated_postfix


class FileUtils:
    @staticmethod
    def fetch_pt_files(data_folder: str, data_file: str):
        """This will fetch all the files in a data folder if
        the specific file does not exit"""
        if not FileUtils.data_exist(data_folder, data_file):
            try:
                file_id = getpass("Enter data folder id:")
                print("Downloading data files ...")
                gdown.download_folder(
                    id=file_id, quiet=True, use_cookies=False, output=data_folder
                )
                print("Downloading process completed")
            except Exception as e:
                print(
                    f"Error: Something went wrong downloading data files. Error {e}.")
                raise

        else:
            print(
                f"Info: Data file exists @ {data_folder}/{data_file}.pt, delete it if you want to fetch it again"
            )

    @staticmethod
    def fetch_checkpoint_files(work_folder: str, checkpoint_file: str, direction: str):
        """This will fetch all the files in a data folder if
        the specific file does not exit"""
        model_folder = f"{work_folder}/saved_best"
        if not FileUtils.data_exist(model_folder, checkpoint_file):
            try:
                file_id = getpass(
                    f"Enter '{direction.capitalize()}' models folder id:")
                print("Downloading model files ...")
                gdown.download_folder(
                    id=file_id, quiet=True, use_cookies=False, output=model_folder
                )
                print("Downloading process completed")
            except Exception as e:
                print(
                    f"Error: Something went wrong downloading data files. Error {e}.")
                raise

        else:
            print(
                f"Info: Model file exists @ {model_folder}/{checkpoint_file}, delete it if you want to fetch it again"
            )

    @staticmethod
    def data_exist(data_folder: str, data_file: str):
        """This checks only for 1 file based on substrate type"""
        return Path(f"{data_folder}/{data_file}").is_file()

    @staticmethod
    def setup_folder_structure(work_folder: str, data_folder: str):
        paths: list[Path] = []
        paths.append(Path(f"{work_folder}/figs"))
        paths.append(Path(f"{work_folder}/saved_best"))
        paths.append(Path(f"{data_folder}"))
        for d in ["direct", "inverse"]:
            paths.append(Path(f"{work_folder}/wandb_logs/{d}/wandb"))
        FileUtils.create_folders(paths)

    @staticmethod
    def create_folders(paths: list[Path]):
        for path in paths:
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"Info: '{path}' exist.")
            except Exception as e:
                print(
                    f"Error: something wrong creating '{path}'. message: {e}.")
                raise

    @staticmethod
    def read_pt_data(data_folder: str, data_file: str) -> Data:
        """Read data from pt file (saved as pytorch tensor)"""
        data_file = Path(f"{data_folder}/{data_file}")
        try:
            data = torch.load(data_file)
        except Exception as e:
            print(
                f"Trouble in loading data file: {data_file}. Error: {e.message}")

        # Set the number of wavelengths
        # config.num_wavelens = data["interpolated_emissivity"].shape[-1]
        return Data(
            laser_params=data["normalized_laser_params"],
            emiss=data["interpolated_emissivity"],
            uids=data["uids"],
            wavelength=data["wavelength"],
        )

    @staticmethod
    def save_best_model(work_folder: str, meta_trainer):
        best_model_path = meta_trainer.model.trainer.checkpoint_callback.best_model_path
        postfix: str = get_dated_postfix(meta_trainer.model)
        dst = Path(f"{work_folder}/saved_best/best_{postfix}.ckpt")

        shutil.copy(best_model_path, dst)
