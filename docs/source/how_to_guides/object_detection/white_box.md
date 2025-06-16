# How to Simulate White Box Attacks

<!-- ```{article-info}
:avatar: 
:avatar-link: 
:avatar-outline: muted
:author: 
:date:
:read-time: 10 min read
:class-container: sd-p-2 article-info-block
``` -->

<!-- ```{note}
The companion Jupyter Notebook can be [found on GitHub here:]](https://github.com/IBM/heart-library).
``` -->

## Introduction

This notebook provides a beginner friendly introduction to using adversarial attacks on image classification as part of
Test & Evaluation of a small benchmark dataset (CIFAR-10). The first attack we will use is Projected Gradient Descent
(PGD), a simple attack based on the model gradients warranted the input. We then visualize the attack and afterwards
constrain the change applied to the image to a fraction of the input image using a patch attack. Computing the
performance under these white-box attacks is a crucial step in T&E.

:::: {grid} 4
:::{grid-item} **Intended Audience:** All T&E Users
:::
:::{grid-item} **Requirements:** Basic Python and Torchvision / ML Skills
:::
:::{grid-item} **Notebook Runtime:** ~2 Minutes
:::
:::{grid-item} **Reading time:** ~10 Minutes
:::
::::

::::{grid} 2

:::{grid-item}
:columns: 8
Before you begin, you will want to make sure that you download the how-to guide's companion
Jupyter notebook. This notebook allows you to follow along in your own environment and interact with the code as you
learn. The code snippets are also included in the documentation, but the notebook is provided for ease of use and to
enable you to try things on your own.
:::

:::{grid-item}
:child-align: center
:columns: 4

<!-- ```{button-link} #
:color: primary
:outline:
Download Companion Notebook {octicon}`download`
``` -->

```{admonition} Coming Soon
:class: important

The companion Jupyter notebook will be available in the code repository in the next HEART release.
```

:::

::::

### Contents

1. Imports and set-up
1. Load data and model
1. Projected Gradient Descent Attack
1. Patch attack
1. Conclusion
1. Next Steps

### Learning Objectives

- How to define a custom model and use CIFAR 10
- How to run an attack with JATIC
- How to inspect images and understand whether they fool the model
- How white box attacks (PGD, Patch attacks) work

## 1. Imports and Set-up

We import all necessary libraries for this tutorial. In this order, we first import general libraries such as numpy,
then load relevant methods from ART. We then load the corresponding HEART functionality and specific torch functions to
support the model. Lastly, we use a command to plot within the notebook.

```python
import numpy as np
import os
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt

#ART imports
from art.utils import load_dataset as load_dataset_art
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.attacks.evasion import AdversarialPatchPyTorch

#HEART imports
from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
from heart_library.metrics import AccuracyPerturbationMetric
from heart_library.attacks.attack import JaticAttack
from heart_library.metrics import HeartAccuracyMetric

#torchvision imports
import torchvision
from torchvision import transforms

%matplotlib inline
```

## 2. Load CIFAR-10 Data and Model for Classification

We load the data, importing only a small part to save compute for this small demonstration. We then define the model and
wrap it as JATIC pytorch classifier.

The data can be replaced as desired by the user - we first define the ten CIFAR labels, specify the subset used in this
notebook (25 images), specify the image size (32 x 32 images) and then safe everything as a modified dataframe.

```python
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset_art('cifar10')
i = 100
x_train = x_train[:100, :].transpose(0, 3, 1, 2).astype('float32')*255
x_test = x_test[:100, :].transpose(0, 3, 1, 2).astype('float32')*255
y_train = y_train[:100, :].astype('float32')
y_test = y_test[:100, :].astype('float32')
```

We then load a custom model which comes with the repository. Most important is that the model has the correct input
shape and is trained to perform decently well on the data. At the bottom, we wrap the model into a
JaticPyTorchClassifier.

```python
path = '../../'

class Model(torch.nn.Module):
        """
        Create model for pytorch.
        Here the model does not use maxpooling. Needed for certification tests.
        """

        def __init__(self):
            super(Model, self).__init__()

            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=(4, 4), dilation=(1, 1), padding=(0, 0), stride=(3, 3)
            )

            self.fullyconnected = torch.nn.Linear(in_features=1600, out_features=10)

            self.relu = torch.nn.ReLU()

            w_conv2d = np.load(
                os.path.join(
                    os.path.dirname(path),
                    "utils/resources/models",
                    "W_CONV2D_NO_MPOOL_CIFAR10.npy",
                )
            )
            b_conv2d = np.load(
                os.path.join(
                    os.path.dirname(path),
                    "utils/resources/models",
                    "B_CONV2D_NO_MPOOL_CIFAR10.npy",
                )
            )
            w_dense = np.load(
                os.path.join(
                    os.path.dirname(path),
                    "utils/resources/models",
                    "W_DENSE_NO_MPOOL_CIFAR10.npy",
                )
            )
            b_dense = np.load(
                os.path.join(
                    os.path.dirname(path),
                    "utils/resources/models",
                    "B_DENSE_NO_MPOOL_CIFAR10.npy",
                )
            )

            self.conv.weight = torch.nn.Parameter(torch.Tensor(w_conv2d))
            self.conv.bias = torch.nn.Parameter(torch.Tensor(b_conv2d))
            self.fullyconnected.weight = torch.nn.Parameter(torch.Tensor(w_dense))
            self.fullyconnected.bias = torch.nn.Parameter(torch.Tensor(b_dense))

        def forward(self, x):
            """
            Forward function to evaluate the model
            :param x: Input to the model
            :return: Prediction of the model
            """
            x = self.conv(x)
            x = self.relu(x)
            x = x.reshape(-1, 1600)
            x = self.fullyconnected(x)
            x = torch.nn.functional.softmax(x, dim=1)
            return x

# Define the network
model = Model()

# Define a loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#wrap classifier
jptc = JaticPyTorchClassifier(
    model=model, loss=loss_fn, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 255),
    preprocessing=(0.0, 255)
)

#plot original image
pred_batch = jptc(x_train[[0]])
plt.imshow(x_train[0].transpose(1,2,0).astype(np.uint8))
_ = plt.title(f'benign classification: {labels[np.argmax(np.stack(pred_batch))]}')
plt.show()
```

```{image} /_static/how-to/wb-1.png
:alt: Frog
```

The above picture shows, in a resolution of 32 x 32 pixels, the image of a frog with the model's classification output
above.

## 3. Define and Run PGD Attack

We are now ready to define the first attack, Projected Gradient Descent (PGD). We then apply the attack to the first
batch in the training data, applying many iterations to ensure that the images fool the classifier. We first plot the
result for two samples and then compute the benign and adversarial performance on a larger batch. From there we observe
that while the initial performance is correct, the final classification is wrong (however we are computing the metric
only on one sample).

When rerunning this notebook, this part can be counted as completed if a correctly classified sample is misclassified.

```python
#define attack
pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=1000, eps=11, eps_step=0.1, targeted=False)

#run attack
x_adv = pgd_attack.generate(x=x_train[[0]], y=y_train[[0]])

#plot adversarial counterpart of image above
pred_batch = jptc(x_adv[[0]])
plt.imshow(x_adv[0].transpose(1,2,0).astype(np.uint8))
_ = plt.title(f'adversarial classification: {labels[np.argmax(np.stack(pred_batch))]}')
plt.show()
```

```text
PGD - Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

```{image} /_static/how-to/wb-2.png
:alt: cat
```

This is the same picture as above, still 32 x 32 resolution, but with a perturbation added that changes the
classification (as depicted above the image).

```python
#define attack, wrap within JATIC counterpart
pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=1000, eps=11, eps_step=0.1, targeted=False)
attack = JaticAttack(pgd_attack, norm=2)

#attack small batch
x_adv, y, metadata = attack(data=x_train[[5]])

pred_batch = jptc(x_train[[5]])
plt.imshow(x_train[5].transpose(1,2,0).astype(np.uint8))
_ = plt.title(f'benign classification: {labels[np.argmax(np.stack(pred_batch))]}')
plt.show()

pred_batch = jptc(np.stack(x_adv)[[0]])
plt.imshow(x_adv[0].transpose(1,2,0).astype(np.uint8))
_ = plt.title(f'adversarial classification: {labels[np.argmax(np.stack(pred_batch))]}')
plt.show()

#attack small batch
x_adv, y, metadata = attack(data=x_train[0:25])

groundtruth_target_batch = [np.argmax(y_train[0:25], axis=1)]
benign_preds_batch = jptc(x_train[0:25])
adversarial_preds_batch = jptc(x_adv)

metric = AccuracyPerturbationMetric(benign_preds_batch, metadata)
metric.update(adversarial_preds_batch, groundtruth_target_batch)
print(metric.compute())
```

```text
PGD - Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

:::: {grid} 2
:::{grid-item}

```{image} /_static/how-to/wb-3.png
:alt: truck
```

:::
:::{grid-item}

```{image} /_static/how-to/wb-4.png
:alt: cat
```

:::
::::

```text
PGD - Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

```python
{'clean_accuracy': 0.56, 'robust_accuracy': 0.16, 'mean_delta': 572.6147}
```

## 4. Patch Attacks

PGD applies perturbations to the entire image. This is not necessarily desired, so next we use an attack that applies a
local perturbation, or a patch, to the image. Analogous to before, we define attack parameters (including patch size and
placement) and optimize the patch. We first plot the patch by itself, before visualizing it on different samples. This
visualization shows us that the patch is not effective on all samples.

When rerunning this notebook, this part can be counted as completed if a correctly classified sample is misclassified.

```python
batch_size = 16
scale_min = 0.3
scale_max = 1.0
rotation_max = 0
learning_rate = 0.1
max_iter = 1000
patch_shape = (3, 10, 10)
patch_location = (0,0)

ap = JaticAttack(AdversarialPatchPyTorch(estimator=jptc, rotation_max=rotation_max, patch_location=patch_location,
                      scale_min=scale_min, scale_max=scale_max, patch_type='circle',
                      learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                      patch_shape=patch_shape, verbose=True, targeted=False))

patched_images, _, metadata = ap(data=x_train)
patch = metadata[0]["patch"]
patch_mask = metadata[0]["mask"]

#plot patch
plt.axis("off")
plt.imshow(((patch) * patch_mask).transpose(1,2,0))
_ = plt.title('Generated Adversarial Patch')
plt.show()
```

```text
Adversarial Patch PyTorch:   0%|          | 0/1000 [00:00<?, ?it/s]
```

```python
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
```

```{image} /_static/how-to/wb-5.png
:alt: patch
```

```python
#plot the images with the patch applied and prediction
preds = jptc(patched_images)
for i, patched_image in enumerate(patched_images[0:5]):
    _ = plt.title(f'''Prediction: {labels[np.argmax(preds[i])]}''')
    plt.imshow(patched_image.transpose(1,2,0).astype(np.uint8))
    plt.show()
```

:::: {grid} 3
:::{grid-item}

```{image} /_static/how-to/wb-6.png
:alt: cat
```

:::
:::{grid-item}

```{image} /_static/how-to/wb-7.png
:alt: dog
```

:::
:::{grid-item}

```{image} /_static/how-to/wb-8.png
:alt: truck
```

:::
::::
:::: {grid} 3
:::{grid-item}

```{image} /_static/how-to/wb-9.png
:alt: frog
```

:::
:::{grid-item}

```{image} /_static/how-to/wb-10.png
:alt: automobile
```

:::
:::{grid-item}

:::
::::

## 5. Conclusion

We have successfully attacked a model with adversarial example that are based on the model's gradients. In the next
steps, we can attempt to defend the computed examples or decrease the knowledge the attacker has by running black box
attacks which are not based on the model's gradients.

## 6. Next Steps

Check out other How-to Guides focusing on:

- [Black-Box Attacks](black_box.md)
- [AutoAttack](auto_attacks.md)
