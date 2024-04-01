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
"""
Utility methods for converting data types to ART compatible versions.
"""
from collections import UserDict
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch as tr
from maite._internals.interop.utils import is_pil_image
from maite.protocols import ArrayLike, HasDataImage, HasDataLabel, is_list_of_type, is_typed_dict
from numpy.typing import NDArray


def process_inputs_for_art(
    data: Union[HasDataImage, Sequence[ArrayLike], ArrayLike], device
) -> Tuple[NDArray, Optional[Any]]:
    """Convert JATIC supported data to ART supported data"""

    images: Any = None
    labels: Any = None

    # If single PIL image, list of one and handle below
    if is_pil_image(data):
        data = [data]

    # If single array, try to convert to tensor and stack and just return
    if not isinstance(data, (dict, UserDict, Sequence)):
        images = tr.as_tensor(data).to(device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return images.cpu().numpy(), None

    # Extract any data out of a dictionary
    labels = None
    if isinstance(data, (dict, UserDict)):
        assert is_typed_dict(data, HasDataImage), "Dictionary data must contain 'image' key."
        images = data["image"]

        if not isinstance(images, Sequence):
            images = [images]

        if is_typed_dict(data, HasDataLabel):
            labels = data["label"]

            if not isinstance(labels, Sequence):
                labels = [labels]

            if isinstance(labels, Sequence) and all(isinstance(label, tr.Tensor) for label in labels):
                labels = [label.cpu().numpy()[0] for label in labels]
    else:
        images = data

    assert isinstance(images, Sequence)

    # If list of PIL images, convert to tensor
    if is_pil_image(images[0]):
        images = [tr.as_tensor(np.array(image, np.float32) / 255).permute((2, 0, 1)).contiguous() for image in images]

        if isinstance(images, Sequence):
            assert isinstance(images[0], tr.Tensor), f"Invalid type {type(images[0])}"
            images = [tr.as_tensor(i).to(device) for i in images]
        else:
            assert isinstance(images, tr.Tensor)
            images = images.to(device)

    # If list of numpy arrays, convert to tensor
    if is_list_of_type(images, np.ndarray):  # type: ignore
        if len(images[0].shape) < 4:
            images = [tr.as_tensor(i).to(device) for i in images]
        else:
            images = tr.as_tensor(images[0]).to(device)

    # If list of numpy arrays
    if is_list_of_type(labels, np.ndarray):  # type: ignore
        if len(labels[0].shape) >= 2:
            labels = labels[0]

    # For list of Tensors, try to stack into a single tensor
    if is_list_of_type(images, tr.Tensor):
        first_item = images[0]
        shape_first = first_item.shape  # type: ignore

        def gen():
            return (shape_first == item.shape for item in images)

        if all(gen()):
            images = tr.stack(images)

    assert is_list_of_type(images, tr.Tensor) or isinstance(images, tr.Tensor)
    if is_list_of_type(images, tr.Tensor):
        images = tr.cat(images).cpu().numpy()
    if isinstance(images, tr.Tensor):
        images = images.cpu().numpy()
    return images, labels
