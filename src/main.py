from config import Config
from trainer_factory import TrainerFactory
from utils import DataUtils, FileUtils


def main(config: Config):
    # TODO: add step to get raw file and create pt file from them (raw2pt)
    # Get pt files (Downloading them if they are missing)
    FileUtils.get_pt_files(config)

    # Load pt file data
    data = DataUtils.read_pt_data(config)

    # 2. forward training_step
    # forward_trainer = TrainerFactory(config, data, "forward")
    # forward_trainer.fit()

    # 3. backward training
    backward_trainer = TrainerFactory(config, data, 'backward')
    backward_trainer.fit()

    # 4. visualization

    return True


if __name__ == "__main__":
    config = Config()
    main(config)
