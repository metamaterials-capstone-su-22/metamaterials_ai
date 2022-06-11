from getpass import getpass
from pathlib import Path

import gdown

from config import Config


class FileUtils:
    @staticmethod
    def get_pt_files(data_path:str, data_file:str):
        """This will fetch all the files in a data folder if
        the specific file does not exit"""
        if not FileUtils.data_exist(data_path, data_file):
            file_id = getpass("Enter data folder id:")
            print("Downloading data files ...")
            gdown.download_folder(
                id=file_id, quiet=True, use_cookies=False, output="local_data"
            )
            print("Downloading process completed")

        else:
            print(
                f"Info: Data file exists @ {data_path}/{data_file}.pt, delete it if you want to fetch it again"
            )

    @staticmethod
    def data_exist(data_path: str, data_file: str):
        """This checks only for 1 file based on substrate type"""
        return Path(f"{data_path}/{data_file}").is_file()

    @staticmethod
    def setup_folder_structure(work_path: str, data_path: str):
        paths : list[Path] = []
        paths.append( Path(f"{work_path}/figs"))
        paths.append( Path(f"{data_path}"))
        for d in ['forward', 'backward']:
            paths.append( Path(f"{work_path}/wandb_logs/{d}/wandb"))
        FileUtils.create_folders(paths)


    @staticmethod    
    def create_folders(paths: list[Path]):
        for path in paths:
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                print(f"Info: '{path}' exist.")
            except Exception as e:
                print(f"Error: something wrong creating '{path}'. message: {e}.")
                raise




