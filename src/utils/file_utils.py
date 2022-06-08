import gdown
from getpass import getpass
from config import Config
from pathlib import Path


class FileUtils():
    def __init__(self, config: Config):
        self.data_path = config.data_path
        self.substrate = config.substrate

    def get_pt_files(self):
        '''This will fetch all the files in a data folder if 
        the specific file does not exit'''
        if not self.data_exist():
            file_id = getpass('Enter data folder id:')
            print('Downloading process completed')
            gdown.download_folder(
                id=file_id, quiet=True, use_cookies=False, output='local_data')
            print('Downloading process completed')

        else:
            print(
                f'Data file exists @ {self.data_path}/{self.substrate}.pt, delete it if you want to fetch it again')

    def data_exist(self):
        '''This checks only for 1 file based on substrate type'''
        data_file = Path(f'{self.data_path}/{self.substrate}.pt')
        return data_file.is_file()
