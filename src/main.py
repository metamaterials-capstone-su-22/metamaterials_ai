from config import Config
from utils import FileUtils, DataUtils


def main(config):
    # Get pt files (Downloading them if they are missing)
    FileUtils.get_pt_files(config)
    
    # Load pt file data
    data = DataUtils.read_pt_data(config)

    # 2. forward training_step
    # 3. backward training
    # 4. visualization

    return True


if __name__ == "__main__":
    config = Config()
    main(config)
