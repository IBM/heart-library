# DoD Tutorial - Part 4: Takeaways and Next Steps

<!-- ```{admonition} Reminder: Download Notebook
:class: important

Remember to download the associated Jupyter notebook to follow along and try the code yourself.  It is located [here](#).
```  -->

```{admonition} Coming Soon
:class: important

The companion Jupyter notebook will be available in the code repository in the next HEART release.
```

## Introduction

::::{grid} 2

:::{grid-item}
:columns: 8
Welcome to Part 4 of the Drone Object Detection Tutorial for HEART. In Part 4, we will review what you learned so far
from each section and provide additional resources for you to review to extend your learning further.
:::

:::{grid-item}
:columns: 4

```{image} /_static/tutorial-drone/amelia.jpg
:alt: Amelia
```

:::

::::

## Part 1: Background, Imports, and Preparation

In Part 1, you met T&E engeineer Amelia and learned about a common use case for HEART - drone object detection. You also
learned about the key features of HEART (Advanced Attack Methods, MAITE Alignment, Defense Mechanisms, and
Exensibility/Open-Source Nature), prerequisities for using HEART, and how to get your environment set up to use HEART.

Notably, you also learned about HEART's "estimators" which are wrappers to put around the model in order to run the
attack evaluations. Furthermore, you were able to produce initial object detector outputs in the form of images with
bounding boxes around certain objects. From here, you could both visually and programmatically inspect for metric
performance.

## Part 2: Attacking the Model

In Part 2, you learned how to use HEART to attack a model using the most powerful white box style attack - the Projected
Gradient Descent attack. This helped Amelia understand the worst case attack scenario when an adversary has access to
the model information. You also learned to use HEART to set the attack parameters for this attack. In the tutorial,
Amelia used the default parameters as a starting point. In the future, she can go back and further tune these parameters
to better characterize the threat profile of the model.

You also learned how to compare model performance between the original and attacked model. You did this via visual
inspection and via analyzing the performance metrics of both models. This provides an initial assessment of the model's
adversarial robustness.

## Part 3: Defending the Model and Advanced Attacks

In Part 3 of the tutorial, you learned about a practical and easy to apply defense, JPEG compression. This is known as a
**"mitigating defense"**. While it helps mitigate the affects of the PGD attack, it is not a **"certified defense"** and
does not have any mathematical guarantees on model robustness.

You were also able to see the application of another advanced attack technique called a **patch attack** that was
applied in an adaptive manner, meaning it was applied knowing that the model was defended with JPEG compression. The
results showed that the patch attack indeed affected model performance negatively and that adaptive attacks are real
threats the exploit further model vulnerabilities beyond what an initial defense can defend against.

## Next Steps and Resources

Now that you have completed the tutorial, make sure to check out the [How-to Guides](/how_to_guides/index.md) referenced
along the way. These guides will provide a more tactical and specific explanation of how to complete the different tasks
that Amelia completed to use HEART. Additionally, the HEART Documentation website offers a number of further resources
to continue to learn about HEART and adversarial robustness including in depth [Explanations](/explanations/index.md) on
concepts related to HEART and other [Reference Materials](/reference_materials/index.md) such as the list of current
models supported by HEART.

We encourage you to now use what you learned and the sample notebook to try to use HEART with your own projects, models,
and datasets!

---

## You Completed the Drone Object Detection Tutorial

Congratulations! You completed Part 4 and the entirety of the Drone Object Detection Tutorial. We look forward to you
using what you learned along with HEART to test and the fortify your models.

::::{grid} 2

:::{grid-item}
:columns: 4

:::

:::{grid-item}
:child-align: end
:columns: 4

```{button-ref} drone_tutorial_3
:color: primary
:expand:
:outline:
:ref-type: doc
Back to Part 3
```

:::

:::{grid-item}
:child-align: end
:columns: 4

```{button-ref} /index
:color: primary
:expand:
:ref-type: doc
Back to Homepage
```

:::

::::
