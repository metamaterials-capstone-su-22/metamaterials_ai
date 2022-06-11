# metamaterials_ai
This repo contains the metamaterials AI Model and training processes.

# How to use Conda
Conda is an open-source package management system and environment management system that runs on Windows, macOS, and Linux. 
One time installation is required. Follow the [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) according your operating system.

Create a virtual environment using Conda

```bash
$ conda create -n meta python=3.10
```

# Pre-run 

You need to modify configuration file `config.py`. Make sure that the `work_path` and `data_path` folders are created before start.

**TODO**: Will add .env file to overload the config instead of manual config file changes.

## Data path
The data files will be read/stored in the location defined by `data_path` in the `config`. The default is `local_data` which is in the project folder. **Note** `local_data` folder is ignored and should not be pushed into the repo.

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

