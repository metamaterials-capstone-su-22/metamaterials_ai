from configparser import ConfigParser
from math import floor

from pydantic import BaseModel


class Config(BaseModel):
    auto_batch_size_adjustment: bool = False
    configs_folder = "configs"
    create_plots = False
    # name of the data file #inconel-shuffled.pt, steel-shuffled.pt
    data_file = "inconel-onehot.pt"
    data_folder: str = "local_data"  # Path to the data folder
    data_portion: float = 1  # Percentage of data being used in the (0 - 1]
    direction: str = "both"  # direct, inverse, both
    direct_arch = "res-ann"  # options 'ann', 'res-ann', 'cnn, 'mlpmixer'
    direct_batch_size: int = None  # 2**9 512
    direct_gamma: float = .1  # schedular gamma
    direct_lr: float | None = None  # leave default to None
    direct_milestones: str | None = None  # '50,100,150'
    direct_num_epochs: int = 1600  # default 1600
    # Default= None: It should be under'{work_folder}/saved_best'
    # = "D-0.9-res-ann-inconel.ckpt"  # Default None
    direct_saved_ckpt: str | None = 'D-1-res-ann-inconel.ckpt'
    inverse_arch = "res-ann"   # options 'ann', 'res-ann', 'cnn, 'mlpmixer'
    inverse_batch_size: int = None  # 2**9 512
    inverse_gamma: float = .1  # schedular gamma
    inverse_lr: float = None  # tune.loguniform(1e-6, 1e-5)
    inverse_milestones: str | None = None  # '50,100,150'
    inverse_num_epochs: int = 2500  # Default 2500
    enable_early_stopper: bool = True  # when 'True' enables early stopper
    # Default= None: It should be under'{work_folder}/saved_best'
    # str | None =  # Default: None
    inverse_saved_ckpt: str | None = 'I-1-res-ann-inconel.ckpt'
    load_direct_checkpoint: bool = True
    load_inverse_checkpoint: bool = True
    num_gpu: int = 1  # number of GPU
    # TODO: Fix num_wavelens be set at load time
    num_wavelens: int | None = 800  # This will be set @ load time. ex. 800
    substrate: str = "inconel"  # options "stainless" , "inconel"
    # use_cache true means to use the .pt file instead of regenerating this
    use_cache: bool = True
    use_direct: bool = True
    should_verify_configs = True  # when true it does some config check before starting
    verbose: bool = True  # When True then will have more log data
    weight_decay = 1e-2  # default for AdamW 1e-2
    # Path to the working folder, checkpoint, graphs, ..
    work_folder: str = "local_work"

    def __init__(self, **data):
        super().__init__(**data)

        # TODO add arg parser if needed
        # args = ArgParser().args

    def load_config_from_file(self):
        if self.should_verify_configs:
            self.verify_config()

        direct_parser = self.get_parser("direct")
        inverse_parser = self.get_parser("inverse")

        self.direct_lr = self.direct_lr or float(
            direct_parser["direct_lr"]) or 1e-3
        self.direct_batch_size = (
            self.direct_batch_size or int(
                direct_parser["direct_batch_size"]) or 2**7
        )
        self.direct_milestones = self.direct_milestones or inverse_parser[
            "direct_milestones"] or '25,50,130,140,150'

        self.inverse_lr = self.inverse_lr or float(
            inverse_parser["inverse_lr"]) or 1e-6
        self.inverse_batch_size = (
            self.inverse_batch_size
            or int(inverse_parser["inverse_batch_size"])
            or 2**7
        )
        self.inverse_milestones = self.inverse_milestones or inverse_parser[
            "inverse_milestones"] or '25,50,130,140,150'

        if self.auto_batch_size_adjustment:
            self.adjust_batch_sizes()

    def adjust_batch_sizes(self):
        self.direct_batch_size = floor(
            self.data_portion * self.direct_batch_size)
        self.inverse_batch_size = floor(
            self.data_portion * self.inverse_batch_size)

    def get_parser(self, direction):
        arch = self.direct_arch if direction == "direct" else self.inverse_arch
        inverse_parser = ConfigParser()
        try:
            inverse_parser.read(f"{self.configs_folder}/{arch}.cfg")
        except:
            print(
                f"Error in reading config file: {self.configs_folder}/{arch}.cfg")
            raise
        return inverse_parser[self.substrate]

    def verify_config(self):
        """Add constraints to configurations here to be checked"""
        if not self.data_file.lower().startswith(self.substrate.lower()):
            print(
                f"Warning: Data file'{self.data_file}' does not match with substrate '{self.substrate}."
            )
            t = input(f"Do you want to continue?")
            if not t == "y":
                raise Exception("Fix the mismatch and re-run!")
