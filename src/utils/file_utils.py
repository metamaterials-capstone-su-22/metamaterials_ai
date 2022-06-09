from getpass import getpass
from pathlib import Path

import gdown

from config import Config


class FileUtils:
    @staticmethod
    def get_pt_files(config):
        """This will fetch all the files in a data folder if
        the specific file does not exit"""
        data_path = config.data_path
        substrate = config.substrate
        if not FileUtils.data_exist(data_path, substrate):
            file_id = getpass("Enter data folder id:")
            print("Downloading data files ...")
            gdown.download_folder(
                id=file_id, quiet=True, use_cookies=False, output="local_data"
            )
            print("Downloading process completed")

        else:
            print(
                f"Data file exists @ {data_path}/{substrate}.pt, delete it if you want to fetch it again"
            )

    @staticmethod
    def data_exist(data_path: str, substrate: str):
        """This checks only for 1 file based on substrate type"""
        data_file = Path(f"{data_path}/{substrate}.pt")
        return data_file.is_file()
