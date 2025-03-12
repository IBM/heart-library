# Quick Start

Installation and setup information can also be found on [GitHub](https://github.com/IBM/heart-library).

To install the latest version of HEART from PyPI, run:

```
$ pip install heart-library
```

To install the latest version of HEART from the heart-library public GitHub, run:

```
$ git clone https://github.com/IBM/heart-library.git
$ cd heart-library
$ pip install .
```

(Optional) Development Environment via Poetry

It may be beneficial for developers to set up an environment from a reproducible source of truth. This environment is useful for developers that wish to work within a pull request or leverage the same development conditions used by HEART contributors. Please follow the [instructions for installation via Poetry](https://github.com/IBM/heart-library/blob/main/poetry_installation.md) within the official HEART GitHub repository.

```
$ conda env create -f environment.yml
$ conda activate heart-env
$ poetry install --all-extras --with dev
```

If conda is not currently installed on your device, instructions to do so can be found [here](https://docs.conda.io/projects/miniconda/en/latest/). If you prefer conda-forge, you can download the installer [here](https://conda-forge.org/).
