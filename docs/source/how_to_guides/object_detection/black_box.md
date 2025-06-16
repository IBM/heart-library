# How to Simulate Black Box Attacks

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

This notebook provides a beginner friendly introduction to using adversarial black-box attacks on image classification
as part of Test & Evaluation of a small benchmark dataset (CIFAR-10). In contrast to white-box attacks that rely on
gradients, black-box attacks approximate the needed gradients based on the output of the model. We apply HopSkipJump,
that does not rely on computing the gradients of the model. As a more practical variant, we show how to use another
black-box attack, the laser beam attack. The changes introduced are here in the shape of a laser beam. To conclude the
notebook, we will show how to calculate clean and attack accuracy. Computing the performance under attack is a crucial
step in T&E.

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
1. HopeSkipJump Attack
1. Laserbeam attack
1. Conclusion
1. Next Steps

### Learning Objectives

- How to define a custom model and use CIFAR 10
- How black-box attacks (HopSkipJump, Laser Beam attack) work

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
from art.attacks.evasion.hop_skip_jump import HopSkipJump

#HEART imports
from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
from heart_library.metrics import AccuracyPerturbationMetric
from heart_library.attacks.evasion import HeartLaserBeamAttack
from heart_library.attacks.attack import JaticAttack
from heart_library.metrics import HeartAccuracyMetric

#torchvision imports
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

%matplotlib inline
```

## 2. Load CIFAR-10 Data and Model for Classification

We load the data, importing only a small part to save compute for this small demonstration. We then define the model and
wrap it as JATIC pytorch classifier.

The data can be replaced as desired by the user - we first define the ten CIFAR labels, specify the subset used in this
notebook (25 images), specify the image size (32 x 32 images) and then safe everything as a modified dataframe.

```python
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

data = load_dataset("cifar10", split="test[0:25]") 

#Transform dataset
IMAGE_H, IMAGE_W = 32, 32

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_H, IMAGE_W)),
    transforms.ToTensor()
])

data = data.map(lambda x: {"image": preprocess(x["img"]), "label": x["label"]}).remove_columns('img')
to_image = lambda x: transforms.ToPILImage()(torch.Tensor(x))
```

We then load a custom model which comes with the repository. Most important is that the model has the correct input
shape and is trained to perform decently well on the data. At the bottom, we wrap the model into JaticPyTorchClassifier.

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

jptc = JaticPyTorchClassifier(
    model=model, loss = torch.nn.CrossEntropyLoss(), input_shape=(3, 32, 32),
    nb_classes=10, clip_values=(0, 1),
)

#plot one image
preds = np.stack(jptc(data))
plt.imshow(np.array(data[0]['image']).transpose(1,2,0))
plt.title(f'Prediction: {labels[np.argmax(preds[0])]}')
plt.show()
```

```{image} /_static/how-to/bb-1.png
:alt: Frog
```

The above picture shows, in a resolution of 32 x 32 pixels, the image of a cat with the model's classification output
above.

## 3. Black Box or HopSkipJump Attack

Although implicit in the code above, the previous two attacks rely on gradient information from the model. It may be
desired from a more practical perspective that model internals are not known to the attacker. We thus run an additional
attack that computes the perturbation purely on the output, not on the model's gradients. As before, we define the
attack, run it, and plot the results with the classification outputs for inspection.

When rerunning this notebook, this part can be counted as completed if a correctly classified sample is misclassified.

```python
#defining and running attack
evasion_attack = HopSkipJump(classifier=jptc, max_iter=100, max_eval=100, init_eval=1, init_size=1, verbose=True)
attack = JaticAttack(evasion_attack)

x_adv, y, metadata = attack(data=data, norm=2)

#visualization
preds = np.stack(jptc(data))
pred_adv = np.stack(jptc(x_adv))

for i in range(5):
    f, ax = plt.subplots(1,2)
    norm_orig_img = np.asarray(data[i]['image'])
    perturbation = np.linalg.norm(norm_orig_img - x_adv[i])

    ax[0].set_title(f'Prediction: {labels[np.argmax(preds[i])]}')
    ax[0].imshow(np.array(data[i]['image']).transpose(1,2,0))
    ax[0].set_xlabel('Original')


    ax[1].set_title(f'$\\ell ^{2}$={perturbation:.5f}\nPrediction adv: {labels[np.argmax(pred_adv[i])]}')
    ax[1].imshow(x_adv[i].transpose(1,2,0))
    ax[1].set_xlabel('Adversarial image')
    plt.show()
```

```text
HopSkipJump:   0%|          | 0/25 [00:00<?, ?it/s]
```

```{image} /_static/how-to/bb-2.png
:alt: cat
```

```{image} /_static/how-to/bb-3.png
:alt: cat
```

```{image} /_static/how-to/bb-4.png
:alt: cat
```

```{image} /_static/how-to/bb-5.png
:alt: cat
```

```{image} /_static/how-to/bb-6.png
:alt: cat
```

## 4. Practical black-box Laserbeam attack

However, the previous black box attack applied, analogous to PGD, changes to the entire image. We thus finally consider
a black box attack that applies perturbations in the form of laser beams. As before, we define the attack, compute the
accuracy decrease (this time on a batch), and plot the resulting adversarial examples for inspection.

When rerunning this notebook, this part can be counted as completed if a correctly classified sample is misclassified.

```python
#define attack 
laser_attack = HeartLaserBeamAttack(jptc, 5, max_laser_beam=(780, 3.14, 32, 32), random_initializations=10)

attack = JaticAttack(laser_attack, norm=2)

#Generate adversarial images
x_adv, y, metadata = attack(data=data)

#Calc clean and robust accuracy
metric = AccuracyPerturbationMetric(jptc(data), metadata)
metric.update(jptc(x_adv), y)
print(metric.compute())
```

```python
{'clean_accuracy': 0.72, 'robust_accuracy': 0.0, 'mean_delta': 12.637125}
```

```python
#Visualize the results
for img in x_adv[:2]:
    plt.title(f'$\\ell ^{2}$={perturbation:.5f}\nPrediction adv: {labels[np.argmax(pred_adv[i])]}')
    plt.imshow(img.transpose(1,2,0))
    plt.show()
```

```{image} /_static/how-to/bb-7.png
:alt: patch
```

```{image} /_static/how-to/bb-8.png
:alt: patch
```

## 5. Conclusion

We have successfully attacked a model with adversarial example that are based only on the model's in/output. In the next
steps, we can test several attacks at once on a model or we can attempt to defend the computed examples.

## 6. Next Steps

Check out other How-to Guides focusing on:

- [White-Box Attacks](white_box.md)
- [AutoAttack](auto_attacks.md)
