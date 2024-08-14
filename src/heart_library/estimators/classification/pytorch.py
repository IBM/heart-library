# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (HEART) Authors 2024
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
This module implements a JATIC compatible ART PyTorchClassifier.
"""
from typing import Any, Sequence

from art.estimators.classification.pytorch import PyTorchClassifier
from maite.protocols import ArrayLike

from heart_library.utils import process_inputs_for_art


class JaticPyTorchClassifier(PyTorchClassifier):  # pylint: disable=R0901,R0901
    """
    JATIC compatible extension of ART core PyTorchClassifier
    """

    def __init__(self, provider: str = "", **kwargs: Any):

        if provider == "huggingface":
            import transformers  # pylint: disable=C0415
            from art.estimators.classification.hugging_face import \
                HuggingFaceClassifierPyTorch  # pylint: disable=C0415

            model = transformers.AutoModelForImageClassification.from_pretrained(kwargs["model"])
            kwargs["model"] = model
            hf_model = HuggingFaceClassifierPyTorch(**kwargs)
            kwargs["model"] = hf_model.model
        elif provider == "timm":
            from art.estimators.classification.hugging_face import \
                HuggingFaceClassifierPyTorch  # pylint: disable=C0415
            from timm import create_model  # pylint: disable=C0415

            model = create_model(
                kwargs["model"],
                pretrained=True,
                num_classes=kwargs["nb_classes"],
            )
            kwargs["model"] = model
            hf_model = HuggingFaceClassifierPyTorch(**kwargs)
            kwargs["model"] = hf_model.model

        super().__init__(**kwargs)

    def __call__(self, data: Sequence[ArrayLike]) -> Sequence[ArrayLike]:

        # convert to ART supported type
        images, _, _ = process_inputs_for_art(data)

        # make prediction
        output = self.predict(images)

        return [output]
