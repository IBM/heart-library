# Drone Object Detection (DoD) Tutorial - Start Here

## Overview

::::{grid} 2

:::{grid-item}
This step-by-step tutorial is designed to help beginner users navigate HEART and understand its
application in securing machine learning models. Throughout this tutorial, we'll walk you through a typical use case,
providing a hands-on experience with HEART's features and functionalities.

Our use case centers around improving the adversarial robustness of a drone object detection model, a critical
application in defense and security. You will track along side our T&E Engineer persona, Amelia, and see how she handles
using HEART for the first time. By following along, you'll learn how to set up your environment, install HEART, and
employ its various tools to identify vulnerabilities, generate adversarial examples, and apply defenses to enhance your
model's security.
:::

:::{grid-item}

```{image} /_static/tutorial-drone/replicator-drone-army.jpg
:alt: Replicator Drone
```

:::

::::

## Intended Audience

This tutorial is generally intended for new users of HEART. Learners are expected to have proficiency with AI and ML
development, specifically with writing and using Python code. The follow list outlines intended audience examples:

::::{grid} 3

:::{grid-item-card} T&E Engineers
Test and evaluation engineers new to HEART and adversarial robustness techniques.
:::

:::{grid-item-card} AI Developers
Individuals interested in strengthening the security of machine learning models in defense and security applications.
:::

:::{grid-item-card} AI Researchers
Researchers and professionals looking to integrate HEART into their workflow for ongoing model evaluation and
improvement.
:::

::::

## Learning Objectives

By the end of this tutorial, you'll have a solid foundation in using HEART and will be well-equipped to apply these
techniques to your own projects. So, let's embark on this journey together, starting with understanding the context and
value of HEART before diving into its installation, setup, and application in our drone object detection use case. We'll
conclude the tutorial by summarizing key takeaways and outlining next steps to help you continue your exploration of
HEART and adversarial robustness.

## Before You Begin

::::{grid} 2

:::{grid-item}
:columns: 8
Before you begin, you will want to make sure that you download the tutorial's Jupyter notebook. This notebook allows
you to follow along in your own environment and interact with the code as you learn. The code snippets are also
included in the documentation, but the notebook is provided for ease of use and to enable you to try things on your own.
:::

:::{grid-item}
:child-align: center
:columns: 4

```{note}
The [Drone Tutorial Companion Notebook](https://github.com/IBM/heart-library/blob/main/notebooks/tutorials/drone_tutorial_companion_notebook.ipynb)
can be downloaded via the HEART public GitHub.
```

<!-- ```{button-link} #
:color: primary
:outline:
Download Tutorial Notebook {octicon}`download`
``` -->

:::

::::

## Let's Get Started

This tutorial is presented in four parts, each building on the next and presenting progressively more complex topics to
the learner.

::::{grid} 4

:::{grid-item-card} Part 1
Background, Imports, and Preparation
:::

:::{grid-item-card} Part 2
Attacking the Model
:::

:::{grid-item-card} Part 3
Defending the Model and Advanced Attacks
:::

:::{grid-item-card} Part 4
Takeaways and Next Steps
:::

::::

::::{grid} 2

:::{grid-item}
:columns: 8

:::

:::{grid-item}
:child-align: end
:columns: 4

```{button-ref} drone_tutorial_1
:color: primary
:expand:
:ref-type: doc
Start Tutorial
```

:::

::::

```{toctree}
:hidden:
drone_tutorial_1.md
drone_tutorial_2.md
drone_tutorial_3.md
drone_tutorial_4.md
```
