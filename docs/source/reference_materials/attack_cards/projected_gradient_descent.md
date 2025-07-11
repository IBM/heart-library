# Projected Gradient Descent

**Attack type:** white box, gradient-based, evasion, digital

**Best for:** models with continuous domain and high-dimensional input data such as images and videos are most
vulnerable to Projected Gradient Descent (PGD) attacks, as there is more opportunity to find a combination of
imperceptible perturbations that are successful.

**Attack summary:** PGD attacks produce adversarial perturbations (slight variations) in an image (or other
approximately continuous domain) that are often difficult to detect with the naked eye. PGD attacks are white-box
attacks, which means that they are only possible when the attacker has access to, and knowledge of, the model
architecture, parameters and gradients. Gradient-based attacks exploit the knowledge of the model’s gradients to
determine how slight changes in the input (pixels in the case of images) influences the output of the model (such as
classification probabilities or object detections). These perturbations are carefully optimized to increase the
likelihood that the input will be misclassified (or objects go undetected) without making significant changes to the
overall image. Perturbations across features can be uniform, small and very hard to detect, which means that the attack
runs a high risk of going undetected.

::::{grid} 2

:::{grid-item-card} Compatibility Considerations

- **Task:** Object detection vs image classification
- **Modality:** HEART currently only supports images, ART supports images and video
- **Data:** Single or three color channel images, of standardized dimensions. Specify pixels in range 0-1 or 0-255,
  matching input data
- **Model:** Must be fully differentiable in order to compute gradients

:::

:::{grid-item-card} Getting Started
To get started with PGD attacks, see the
[Projected Gradient Descent Notebook](https://github.com/IBM/heart-library/blob/main/notebooks/1_get_started_pgd_attack.ipynb),
available via the IBM HEART-library GitHub repository.

For increased relevance to your use case, replace the selected hugging face model with your own model, and the test data
set with a test dataset of your own.
:::

::::

::::{grid} 2

:::{grid-item-card} Interpreting the Results
A model's robustness can be assessed by comparing performance before and
after an attack. For details on how to evaluate model performance and attack effectiveness, see this explanation of
[evaluation metrics](/explanations/evaluation_metrics).
:::

:::{grid-item-card} Remediation Resources

1. Pre-processing
   [mitigation steps](https://github.com/IBM/heart-library/blob/main/notebooks/8_get_started_defenses.ipynb) (image
   compression, spatial smoothing, variance minimization)
1. Defenses like adversarial training (currently supported by ART)

- [Adversarial training example with MNIST](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_training_mnist.ipynb)
- [Adversarial retraining](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_retraining.ipynb)
- [Certified adversarial training](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/certified_adversarial_training.ipynb)

:::

::::

<!-- markdownlint-disable MD013 -->
```{eval-rst}
.. grid:: 1
  :gutter: 1
  :margin: 0

  .. grid-item-card::  Scalability

    The examples of time and compute requirements below cover a variety of models and datasets to guide users' expectations. This data can be used for resource planning for model testing and evaluation (T&E).

    .. csv-table::
      :file: ../../_static/pgd_scalability_metrics.csv
      :widths: 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
      :header-rows: 1
      :class: longtable

    .. raw:: html

      <iframe allowtransparency="true" src="../../_static/pgd_plot_memory.html" height="450px" width="100%"></iframe>
      <iframe allowtransparency="true" src="../../_static/pgd_plot_duration.html" height="450px" width="100%"></iframe>

```

```{eval-rst}
.. grid:: 2
  :gutter: 1
  :margin: 0

  .. grid-item-card::  What could go wrong?

    - Model and input data not compatible --> see 'Compatibility considerations' above.
    - Model is overfit --> won't produce useful gradient information
    - Wrong hyperparameters --> iterations must be enough for attack to converge
    - Landed on false local minimum, no adversarial example present --> modify loss function
    - Model not differentiable or gradient direction doesn't minimize loss function --> loss function must be appropriate to model
    - *Last* sample of attack path returned, not adversarial --> have optimization algorithm return *best* attack path sample

    For more information on causes of attack failure, see Carlini's `Indicators of Attack Failure <https://arxiv.org/pdf/2106.09947>`_ and Tramer's `On Adaptive Attacks to Adversarial Example Defenses <https://proceedings.nips.cc/paper/2020/file/11f38f8ecd71867b42433548d1078e38-Paper.pdf>`_.

  .. grid-item-card::  More Resources

    - Similar attacks:
        
      - PGD is just one type of gradient-based attack. For more information on others, see `article <https://securing.ai/ai-security/gradient-based-attacks/>`_.
      - Adversarial Patch attacks can be applied in similar circumstances as PGD attacks. For more information see the `Getting Started with Adversarial Patch notebook <https://github.com/IBM/heart-library/blob/main/notebooks/4_get_started_adversarial_patch.ipynb>`__, available via the HEART-library GitHub repository.

    - Further reading:
        
      - `Adversarial Robustness Toolbox v1.0.0 <https://arxiv.org/abs/1807.01069>`_
      - `Adversarial Robustness Toolbox repo (v1.18.0+) <https://github.com/Trusted-AI/adversarial-robustness-toolbox>`_ and related `discussions <https://github.com/Trusted-AI/adversarial-robustness-toolbox/discussions>`_

```
<!-- markdownlint-enable MD013 -->

For more information on which attacks are relevant in which conditions, please see
{doc}`HEART's Adversarial Evaluation Pathways <../evaluation_pathways>`.
