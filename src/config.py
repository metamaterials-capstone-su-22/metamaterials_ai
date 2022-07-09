from distutils.command.config import config
from pydantic import BaseModel
from configparser import ConfigParser


class Config(BaseModel):
    inverse_arch = "cnn"  # options 'MLPMixer', 'resnet1d','ann', 'cnn,
    inverse_batch_size: int = None  # 2**9 512
    inverse_lr: float = None  # tune.loguniform(1e-6, 1e-5)
    inverse_num_epochs: int = 2500  # Default 2500
    create_plots = False
    data_file = "stainless-steel-revised-shuffled.pt"  # name of the data file
    data_folder: str = "local_data"  # Path to the data folder
    data_portion: float = 1  # Percentage of data being used in the [.01 - 1]
    direction: str = "both"  # direct, inverse, both
    direct_arch = "cnn"  # options 'MLPMixer', 'resnet1d','ann', 'cnn,
    direct_batch_size: int = None  # 2**9 512
    direct_lr: float | None = None  # leave default to None
    direct_num_epochs: int = 1600  # default 1600
    enable_early_stopper: bool = True  # when 'True' enables early stopper
    load_direct_checkpoint: bool = True
    load_inverse_checkpoint: bool = False
    num_gpu: int = 1  # number of GPU
    # TODO: Fix num_wavelens be set at load time
    num_wavelens: int | None = 800  # This will be set @ load time. ex. 800
    substrate: str = "stainless"  # options "stainless" , "inconel"
    # use_cache true means to use the .pt file instead of regenerating this
    use_cache: bool = True
    use_direct: bool = True
    should_verify_configs = True  # when true it does some config check before starting
    weight_decay = 1e-2  # default for AdamW 1e-2
    # Path to the working folder, checkpoint, graphs, ..
    work_folder: str = "local_work"

    def __init__(self, **data):
        super().__init__(**data)
        if self.should_verify_configs:
            self.verify_config()
        parser = ConfigParser()
        parser.read(
            f'src/configs/{self.direct_arch}.cfg')
        configs = parser[self.substrate]

        self.direct_lr = self.direct_lr or float(configs['direct_lr']) or 1e-3
        self.direct_batch_size = self.direct_batch_size or int(
            configs['direct_batch_size']) or 2**7
        self.inverse_lr = self.inverse_lr or float(
            configs['inverse_lr']) or 1e-6
        self.inverse_batch_size = self.inverse_batch_size or int(
            configs['inverse_batch_size']) or 2**7

        # TODO add arg parser if needed
        # args = ArgParser().args

    def verify_config(self):
        """Add constraints to configurations here to be checked"""
        if not self.data_file.lower().startswith(self.substrate.lower()):
            print(
                f"Warning: Data file'{self.data_file}' does not match with substrate '{self.substrate}."
            )
            t = input(f"Do you want to continue?")
            if not t == "y":
                raise Exception("Fix the mismatch and re-run!")
