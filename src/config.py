from pydantic import BaseModel


class Config(BaseModel):
    inverse_arch = "cnn"  # options 'MLPMixer', 'resnet1d','ann', 'cnn,
    inverse_batch_size: int = 2**7  # 2**9 512
    inverse_lr: float = 1e-6  # tune.loguniform(1e-6, 1e-5),
    inverse_num_epochs: int = 2500  # Default 2500
    create_plots = False
    data_file = "inconel-revised-raw-shuffled.pt"  # name of the data file
    data_folder: str = "local_data"  # Path to the data folder
    direction: str = "both"  # direct, inverse, both
    enable_early_stopper: bool = True  # when 'True' enables early stopper
    direct_arch = "ann"  # options 'MLPMixer', 'resnet1d','ann', 'cnn,
    direct_batch_size: int = 2**7  # 2**9 512
    # leave default to None. tune.loguniform(1e-7, 1e-4),
    direct_lr: float | None = None
    direct_num_epochs: int = 1600  # default 1600
    load_direct_checkpoint: bool = False
    load_inverse_checkpoint: bool = False
    num_gpu: int = 1  # number of GPU
    # TODO: Fix num_wavelens be set at load time
    num_wavelens: int | None = 800  # This will be set @ load time. ex. 800
    substrate: str = "inconel"  # options "stainless" , "inconel"
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

        if not self.direct_lr:
            self.direct_lr = 1e-5 if self.substrate == "inconel" else 1e-3

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
