# Patch Attack

**Attack type:** white-box (supported by HEART), black-box (currently supported by ART), evasion, digital or physical. For more information on types of patch attack see test {doc}`these more detailed explanations <../../explanations/PatchDocumentation>`.

**Best for:** patch attacks are localized and unbounded, making them easy to transfer to the physical world (while remaining applicable in the digital space).

**Attack summary:** Patch attacks are carried out by adding an object to an image that degrades the results of a visual model ingesting that image, either producing the wrong classification, or failing to detect a relevant object within the image. Adversarial patches can be created with access to only the model's output, and are not norm-bound or specific to a single image. Patch attacks are highly versatile and can be implemented both digitally and physically.

::::{grid} 2

:::{grid-item-card} Compatibility considerations
- **Task:** Object detection vs image classification
- **Modality:** HEART currently only supports images, ART supports images and video
- **Data:** Single or three color channel images, of standardized dimensions. Specify pixels in range 0-1 or 0-255, matching input data
- **Model:** Computer vision model
:::

:::{grid-item-card} Getting started
To get started with Patch attacks, see the :ref:`patch-notebook-label` notebook, available [here](https://github.com/IBM/heart-library/blob/main/notebooks/4_get_started_adversarial_patch.ipynb).

For increased relevance to your use case, replace the selected hugging face model with your own model, and the test data set with a test dataset of your own.
:::

::::

::::{grid} 2

:::{grid-item-card} Interpreting the results
A model's robustness can be assessed by comparing performance before and after an attack. For details on how to evaluate model performance and attack effectiveness, see this explanation of [evaluation metrics](/explanations/evaluation_metrics).
:::

:::{grid-item-card} Remediation resources
1. Pre-processing [mitigation steps](https://github.com/IBM/heart-library/blob/main/notebooks/8_get_started_defenses.ipynb) (image compression, spatial smoothing, variance minimization)
2. Defenses like adversarial training (currently supported by ART)

- [Adversarial training example with MNIST](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_training_mnist.ipynb)

- [Adversarial retraining](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_retraining.ipynb)

- [Certified adversarial training](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/certified_adversarial_training.ipynb)
:::

::::



```{eval-rst}
.. grid:: 1
    :gutter: 1
    :margin: 0

    .. grid-item-card::  Scalability

        The examples of time and compute requirements below cover a variety of models and datasets to guide users' expectations. These data can be used for resource planning for model testing and evaluation (T&E).

        .. csv-table::
           :file: ../../_static/patch_scalability_metrics.csv
           :widths: 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
           :header-rows: 1
           :class: longtable

        .. raw:: html

           <iframe allowtransparency="true" src="../../_static/patch_plot_memory.html" height="450px" width="100%"></iframe>
           <iframe allowtransparency="true" src="../../_static/patch_plot_duration.html" height="450px" width="100%"></iframe>

```

```{eval-rst}
.. grid:: 2
    :gutter: 1
    :margin: 0

    .. grid-item-card::  What could go wrong?


        - Model and input data not compatible --> see 'Compatibility considerations' above

        - Patch may be too easily detected

        - Incorrect size, shape, or placement of the patch relative to the original image

        - [in physical patch use] Changes in lighting or object orientation can decrease effectiveness

        For more information on causes of attack failure, see Carlini's `Indicators of Attack Failure <https://arxiv.org/pdf/2106.09947>`_ and Tramer's `On Adaptive Attacks to Adversarial Example Defenses <https://proceedings.nips.cc/paper/2020/file/11f38f8ecd71867b42433548d1078e38-Paper.pdf>`_.


    .. grid-item-card::  More resources

        - Similar attacks:

          - A second patch attack notebook for object detection can be found `here <https://github.com/IBM/heart-library/blob/main/notebooks/6_adversarial_patch_for_object_detection.ipynb>`__.

          - Other physically realizable attacks include `adversarial laser beam <https://arxiv.org/abs/2103.06504>`_.
        - Further reading:

          - `Adversarial Robustness Toolbox v1.0.0 <https://arxiv.org/abs/1807.01069>`_

          - `Adversarial Robustness Toolbox repo (v1.18.0+) <https://github.com/Trusted-AI/adversarial-robustness-toolbox>`_ and related `discussions <https://github.com/Trusted-AI/adversarial-robustness-toolbox/discussions>`_

```

For more information on which attacks are relevant in which conditions, please see {doc}`HEART's Adversarial Evaluation Pathways <../evaluation_pathways>`.
