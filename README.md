# Hardened Extension of the Adversarial Robustness Toolbox (HEART) 

![Static Badge](https://img.shields.io/badge/python-3.9%20--%203.10-blue "Python 3.9 - 3.10 version support.")

HEART is a Python extension library for Machine Learning Security that builds on the popular Adversarial Robustness algorithms within the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox). The extension library allows the user to leverage core ART algorithms while providing additional benefits to AI Test & Evaluation (T&E) engineers. HEART documentation can be found [here](https://heart-library.readthedocs.io/). 

- Support for T&E of models for Department of Defense use cases 
- Alignment to [MAITE](https://github.com/mit-ll-ai-technology/maite) protocols for seamless T&E workflows
- Essential subset of adversarial robustness methods for targeted AI security coverage 
- Quality assurance of model assessments in the form of metadata 
- In-depth support for users based on codified T&E expert experience in form of guides and examples
- Front-end application for low-code users: HEART Gradio Application 

# Installation

### From Python Packaging Index (PyPI)

To install the latest version of HEART from PyPI, run:

```shell
pip install heart-library
```

### From IBM GitHub Source

To install the latest version of HEART from the [heart-library public GitHub](https://github.com/IBM/heart-library), run:

```shell
git clone https://github.com/IBM/heart-library.git
cd heart-library
pip install .
```

### (Optional) Development Environment via Poetry

In some cases, it may be beneficial for developers to set up an environment from a reproducible source of truth.  This environment is useful for developers that wish to work within a pull request or leverage the same development conditions used by HEART contributors.  Please follow the instructions for installation via Poetry within the official HEART repository:

- [Poetry Installation Instructions](https://github.com/IBM/heart-library/blob/main/poetry_installation.md)

# Getting Started With HEART

IBM has published a catalog of notebooks designed to assist developers of all skill levels with the process of getting started utilizing HEART in their AI T&E workflows.  These Jupyter notebooks can be accessed within the official heart-library GitHub repository:

- [HEART Jupyter Notebooks](https://github.com/IBM/heart-library/tree/main/notebooks)

# HEART Modules

The HEART library is organized into three primary modules: attacks, estimators, and metrics.

### heart_library.attacks

> The HEART attacks module contains implementations of attack algorithms for generating adversarial examples and evaluating model robustness.

### heart_library.estimators

> The HEART estimators module contains classes that wrap and extend the evaluated model to make it compatible with attacks and metrics.

### heart_library.metrics

> The HEART metrics module implements industry standard, commonly-used T&E metrics for model evaluation.
