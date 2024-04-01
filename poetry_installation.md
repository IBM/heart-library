# HEART Development Environment Setup Using Poetry

These installation instructions outline how to set up a reproducible environment for HEART, utilizing the `poetry.lock` file that has been published within the heart-library repository.

### Why would I want to set up a HEART environment using Poetry?

**As an AI T&E Developer:**
> As a developer, you might wish to use the same environment conditions that HEART contributors utilized during their development process.  Installing a solved dependency tree via `poetry.lock` ensures this consistency.

**As a developer who has initiated a HEART Pull Request:**
> As a developer that is working within a pull request, it may be advantageous to leverage unit testing within the HEART repository.  In this scenario, one should ensure that the development environment being used aligns with the solved dependency tree utilized by HEART developers defined in `poetry.lock`.

### Prerequisites

**Python**
> To view all supported versions of Python, please review the [pyproject.toml](https://github.com/IBM/heart-library/blob/main/pyproject.toml) file located within the heart-library repository.  The `[tool.poetry.dependencies]` metadata section defines all Python versions that are currently supported.  The Conda environment contained within `environment.yml` will automatically install a supported version of Python.

**Poetry**
> The HEART development environment uses Poetry for dependency management.  The reproducible environment defined via `poetry.lock` represents the set of dependencies (including developer dependencies) for a given release of HEART.  When utilizing the Conda environment defined in `environment.yml`, an officially supported version of Poetry will be automatically installed.  However, if you wish to install Poetry manually, please install the latest version in your virtual environment of choice via PyPI:  `pip install poetry`

**Conda**
> Conda is the preferred virtual environment manager when setting up a development environment for HEART.  This is not mandatory, but the directions in this document assume that Conda is utilized.  If you need help installing Conda, the official Conda installation documentation is available via: https://conda.io/projects/conda/en/latest/user-guide/install/index.html


# Installation Instructions

### Clone the HEART Repository from IBM GitHub

```shell
git clone https://github.com/IBM/heart-library.git
```

### Create the HEART Conda Virtual Environment
> The conda virtual environment defined within `environment.yml` installs officially supported versions of both Python and Poetry.  If you elect to use a different virtual environment, you will need to ensure that the version of Python used is compatible with HEART and that Poetry is successfully installed using PyPI.


```shell
conda env create -f environment.yml
```

### Activate the HEART Conda Virtual Environment
```shell
conda activate heart-env
```

### Install All HEART Dependencies Using Poetry

**With Development Dependencies:**
```shell
poetry install --all-extras --with dev
```

**or**

**Without Development Dependencies:**
```shell
poetry install --all-extras
```
