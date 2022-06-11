# metamaterials_ai
This repo contains the metamaterials AI Model and training processes.

# How to use Conda
Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux. 
One time installation is required. Follow the [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) according your operating system.

Create a virtual environment using Conda 

```bash
$ conda create -n meta python=3.10
```
In the above command `meta` is the name of the environemt.

# Pre-run 

You need to modify configuration file `config.py`. Make sure that the `work_path` and `data_path` folders are created before start.

**TODO**: Will add .env file to overload the config instead of manual config file changes.

## Data path
The data files will be read/stored in the location defined by `data_path` in the `config`. 

The default is `local_data` which is in the project folder. This folder is ignored by git and will not be pushed into the repo.

At the first run if the data file is missing in the path defined in the cofig, the code will fetch the data. It will prompt user for the id of the file. If the files already exist it will not download them.

## Work path
The work folder is where the logs and checkpoints will be stored. the location is defined by `work_folder` in the `config`. The default is `local_work`. Make sure the folder pointed exists. **Note** `local_work` folder is ignored and should not be pushed into the repo.


# How to use Poetry
Poetry can be used to simplify dependency management.

## Poetry installation 
One time installation is required. Follow the [instruction](https://python-poetry.org/docs/master/) according your operating system.

## Install packages using poetry

When packages are added by poetry the list of them will be tracked and they can be installed using the following command

Note: to this work you need to have an environment with the correct version of python (v3.10 for this project) created.

```bash
poerty install
```
## Run using poetry
Assume you are in the `root` folder of the project

```bash
$ poetry run python src/main.py
```

## Add dependency using Poetry

```bash
poetry add <package_name>

# example
poetry add numpy
```
# Config file
The settings are configurable through the config file.

**NOTE** TODO need to add .env overide.

```python
# config.py
class Config(BaseModel):
    backward_batch_size: int = 1000  # 2**9
    backward_lr: float = 1e-6  # tune.loguniform(1e-6, 1e-5),
    backward_num_epochs: int = 2  # 2500
    create_plots = False
    data_file = "stainless_steel.pt" # name of the data file 
    data_path: str = "local_data"  # Path to the data folder
    forward_batch_size: int = 1000  # 2**9
    forward_lr: float = 1e-6  # tune.loguniform(1e-7, 1e-4),
    forward_num_epochs: int = 2  # 1600
    load_forward_checkpoint: bool = False
    load_backward_checkpoint: bool = False
    model_arch = "MLPMixer"  # options 'MLPMixer'
    num_gpu: int = 1  # number of GPU
    # TODO: Fix num_wavelens be set at load time
    num_wavelens: int | None = 800  # This will be set @ load time. ex. 800
    substrate: str = "stainless_steel"  # options "stainless_steel" , "inconel"
    use_cache: bool = (
        True  # Use the pt file instead of reading from the google drive or mongadb
    )
    use_forward: bool = True
    # Path to the working folder, checkpoint, graphs, ..
    work_path: str = "local_work"
```
## Data file
Refer to the Pre-run section to read about where data will be stored.
The current version only tranis based on one substrate. 

