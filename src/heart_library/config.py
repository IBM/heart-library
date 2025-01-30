# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This module loads and provides configuration parameters for HEART"""

import json
import logging
import os

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES

HEART_NUMPY_DTYPE = np.float32
HEART_DATA_PATH: str

# --------------------------------------------------------------------------------------------- DEFAULT PACKAGE CONFIGS

_folder = os.path.expanduser("~")
if not os.access(_folder, os.W_OK):  # pragma: no cover
    _folder = "/tmp"  # noqa S108
_folder = os.path.join(_folder, ".heart")


def set_data_path(path: str) -> None:
    """Set the path for HEART's data directory (HEART_DATA_PATH).

    Args:
        path (str): data path.

    Raises:
        OSError: if path cannot be read from.
    """
    expanded_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(expanded_path, exist_ok=True)
    if not os.access(expanded_path, os.R_OK):  # pragma: no cover
        raise OSError(f"path {expanded_path} cannot be read from")
    if not os.access(expanded_path, os.W_OK):  # pragma: no cover
        logger.warning("path %s is read only", expanded_path)

    global HEART_DATA_PATH
    HEART_DATA_PATH = expanded_path
    logger.info("set HEART_DATA_PATH to %s", expanded_path)


# Load data from configuration file if it exists. Otherwise create one.
_config: dict = {}
_config_path = os.path.expanduser(os.path.join(_folder, "config.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path, encoding="utf8") as _f:
            _config = json.load(_f)

            # Since renaming this variable we must update existing config files
            if "DATA_PATH" in _config:  # pragma: no cover
                _config["HEART_DATA_PATH"] = _config.pop("DATA_PATH")
                try:
                    with open(_config_path, "w", encoding="utf8") as _f:
                        _f.write(json.dumps(_config, indent=4))
                except OSError:
                    logger.warning("Unable to update configuration file", exc_info=True)

    except ValueError:  # pragma: no cover
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:  # pragma: no cover
        logger.warning("Unable to create folder for configuration file.", exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {"HEART_DATA_PATH": os.path.join(_folder, "data")}

    try:
        with open(_config_path, "w", encoding="utf8") as _f:
            _f.write(json.dumps(_config, indent=4))
    except OSError:  # pragma: no cover
        logger.warning("Unable to create configuration file", exc_info=True)

if "HEART_DATA_PATH" in _config:  # pragma: no cover
    set_data_path(_config["HEART_DATA_PATH"])
