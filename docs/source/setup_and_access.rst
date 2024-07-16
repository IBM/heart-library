Setup and Access
================


Access for low code users
-------------------------

Low-code front-end application available at `HEART Gradio Application <https://huggingface.co/spaces/CDAO/HEART-Gradio>`_.


Setup for high code users
-------------------------

Full installation and setup information can be found on `GitHub <https://github.com/IBM/heart-library>`_.


To install the latest version of HEART from PyPI, run:

.. code-block::

      $ pip install heart-library



To install the latest version of HEART from the heart-library public GitHub, run:

.. code-block::

      $ git clone https://github.com/IBM/heart-library.git
      $ cd heart-library
      $ pip install .



(Optional) Development Environment via Poetry
It may be beneficial for developers to set up an environment from a reproducible source of truth.  This environment is useful for developers that wish to work within a pull request or leverage the same development conditions used by HEART contributors.  Please follow the `instructions for installation via Poetry <https://github.com/IBM/heart-library/blob/main/poetry_installation.md>`_ within the official HEART GitHub repository. 

.. code-block::

      $ conda env create -f environment.yml
      $ conda activate heart-env
      $ poetry install --all-extras --with dev

If conda is not currently installed on your devices, instructions to do so can be found `here <https://docs.conda.io/projects/miniconda/en/latest/>`_


IBM has published a catalog of notebooks designed to assist developers of all skill levels with the process of getting started utilizing HEART in their AI T&E workflows.  These Jupyter notebooks can be accessed within the official heart-library GitHub repository: `<https://github.com/IBM/heart-library/tree/main/notebooks>`_
