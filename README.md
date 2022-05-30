# metamaterials_ai
This repo contains the metamaterials AI Model and training processes.

# How to use Conda
Create a virtual environment using Conda

```bash
$ conda create -n meta python=3.10
```

# How to use Poetry
Poetry can be used to simplify dependency management.

## Poetry installation 
One time installation is required. Follow the [instrunction](https://python-poetry.org/docs/#installation) according your operating system.

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

