# Data module for the RICO project

This repository contains the data module for the RICO project from SINTEF. The module is written in Python and is intended to be used in conjunction with the RICO dataset. The module is intended to be used for research purposes.

## Installation as a package

The module can be installed using pip:

```bash
pip install git+https://github.com/Zachari-THIRY/rico-data-module
```
Depending on your system, you may need to use pip3 instead of pip.
Dependencies are listed in the environment.yml file. However, they aren't installed with pip. Please make sure to install them manually (or use the provided environment.yml file to create a conda environment).

Dependencies : numpy, pandas, matplotlib, torch, tables

The module has been tested with PyTorch 2.1.0, but should work with other versions as well.


## Contributing or installation from source

Instructions on how to contribute to the project, for example:

```bash
git clone https://github.com/Zachari-THIRY/rico-data-module
cd rico-data-module
conda env create -f environment.yml
```
This will create a conda environment with the name rico_data. To activate the environment, run the following command:

```bash
conda activate rico_data
```
Note that the requirements are minimal, and should seemlessly integrate with most PyTorch development environments.
Feel free to fork the project and make pull requests !

## Usage/Examples

Instructions on how to use module are provided within the module itself. The module is intended to be used in conjunction with the RICO dataset, which can be downloaded from the following link: (link unavailable yet ...)
Walkthtough notebooks are provided in the notebooks folder.

## Authors

- **Zachari THIRY** - *Initial work* - [zachari](https://github.com/Zachari-THIRY)

## License

Each file provided in this repository is licensed under the _ADD LICENCE_ license.