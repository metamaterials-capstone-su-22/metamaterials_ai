from config import Config
from utils import FileUtils


def main(config):
    # 1. read pt files from google drive
    #   1.1 download file
    # 2. forward training_step
    # 3. backward training
    # 4. visualization
    file_utils= FileUtils(config)
    file_utils.get_pt_files()

    return True


if __name__ == "__main__":
    config = Config()
    main(config)
