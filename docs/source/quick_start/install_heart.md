Installation and Setup
======================

This presents HEART users with a guide to the initial installation and setup of the HEART library.

```{note}
The installation and setup information can also be found on [GitHub](https://github.com/IBM/heart-library).
```

## PyPI Installation

To install the latest version of HEART from PyPI, run:

```
pip install heart-library
```

## Install Latest Version from GitHub

To install the latest version of HEART from the heart-library public GitHub, run:

```
git clone https://github.com/IBM/heart-library.git
cd heart-library
pip install .
```

## (Optional) Development Environment via Poetry

It may be beneficial for developers to set up an environment from a reproducible source of truth. This environment is useful for developers that wish to work within a pull request or leverage the same development conditions used by HEART contributors. Please follow the instructions for installation via Poetry within the official HEART GitHub repository.

```
conda env create -f environment.yml
conda activate heart-env
poetry install --all-extras --with dev
```

If conda is not currently installed on your devices, instructions to do so can be found [here](https://docs.conda.io/projects/miniconda/en/latest/)
