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
as part of Test & Evaluation of a small benchmark dataset based on drone imagery. In contrast to white-box attacks that
rely on gradients, black-box attacks approximate the needed gradients based on the output of the model. We apply
HopSkipJump, that does not rely on computing the gradients of the model. As a more practical variant, we show how to use
another black-box attack, the laser beam attack. The changes introduced are here in the shape of a laser beam. To
conclude the notebook, we will show how to calculate clean and attack accuracy. Computing the performance under attack
is a crucial step in T&E.

:::: {grid} 5
:::{grid-item} **Intended Audience:** All T&E Users
:::
:::{grid-item} **Requirements:** Basic Python and Torchvision / ML Skills
:::
:::{grid-item} **Notebook Runtime:** Full run of the notebook: <20 minutes
:::
:::{grid-item} **Reading time:** ~10 Minutes
:::
:::{grid-item} **Order of Completion:** 3. and 4. can be done in any order or independently.
:::
::::

::::{grid} 2

:::{grid-item}
:columns: 8
Before you begin, you will want to make sure that you download the how-to guide's companion Jupyter notebook.  This
notebook allows you to follow along in your own environment and interact with the code as you learn.  The code snippets
are also included in the documentation, but the notebook is provided for ease of use and to enable you to try things on
your own.
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
1. Targeted Black-box Attack
1. Conclusion
1. Next Steps

### Learning Objectives

- How to define a custom model and a drone imagery dataset
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
from art.attacks.evasion.hop_skip_jump import HopSkipJump

#HEART imports
from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
from heart_library.metrics import AccuracyPerturbationMetric
from heart_library.attacks.evasion import HeartLaserBeamAttack
from heart_library.attacks.attack import JaticAttack

#torchvision imports
import torchvision
from torchvision import transforms

%matplotlib inline
```

## 2. Load Drone Data and Model for Classification

We load the data, importing only a small part to save compute for this small demonstration. We then define the model and
wrap it as JATIC pytorch classifier.

The data can be replaced as desired by the user - we first define the six labels, specify the subset used in this
notebook (10 images), specify a consistent size and upscale the data to 224 x 224 pixels and then wrap everything as a
modified dataframe.

```python
classes = {
    0:'Building',
    1:'Construction Site',
    2:'Engineering Vehicle',
    3:'Fishing Vessel',
    4:'Oil Tanker',
    5:'Vehicle Lot'
}

data = load_dataset("CDAO/xview-subset-classification", split="test[0:15]")
idx = 3
plt.title(f"Prediction: {classes[data[idx]['label']]}")
plt.imshow(data[idx]['image'])

'''
Transform dataset
'''
IMAGE_H, IMAGE_W = 224, 224

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_H, IMAGE_W)),
    transforms.ToTensor()
])
dataT = data.map(lambda x: {"image": preprocess(x["image"]), "label": x["label"]})
to_image = lambda x: transforms.ToPILImage()(torch.Tensor(x))

sample_data = torch.utils.data.Subset(dataT, range(10))
```

```{image} /_static/how-to/ic/bb-1.png
:alt: Building
```

We then load a custom model which comes with the repository. Most important is that the model has the correct input
shape and is trained to perform decently well on the data. At the bottom, we wrap the model into JaticPyTorchClassifier.

```python
model = torchvision.models.resnet18(False)
num_ftrs = model.fc.in_features 
model.fc = torch.nn.Linear(num_ftrs, len(classes.keys())) 
model.load_state_dict(torch.load('../../../utils/resources/models/xview_model.pt'))
#_ = model.eval()

'''
Wrap the model
'''
jptc = JaticPyTorchClassifier(
    model=model, loss = torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224),
    nb_classes=len(classes), clip_values=(0, 1)
)

#plot original image
pred_batch = jptc(sample_data)
plt.imshow(data[1]['image'])
_ = plt.title(f'benign classification: {classes[np.argmax(np.stack(pred_batch[1]))]}')
plt.show()
```

```{image} /_static/how-to/ic/bb-2.png
:alt: building
```

## 3. Black Box or HopSkipJump Attack

White-box attacks rely on gradient information from the model. It may be desired from a more practical perspective that
model internals are not known to the attacker. We thus run an additional attack that computes the perturbation purely on
the output, not on the model's gradients. As before, we define the attack, run it, and plot the results with the
classification outputs for inspection.

When rerunning this notebook, this part can be counted as completed if a correctly classified sample is misclassified.

```python
#defining and running attack
evasion_attack = HopSkipJump(
    classifier=jptc, max_iter=100, max_eval=100,
    init_eval=1, init_size=1, verbose=True, targeted=False,
)

attack = JaticAttack(evasion_attack)

x_adv, y, metadata = attack(data=sample_data, norm=2)

#visualization
preds = np.stack(jptc(sample_data))
pred_adv = np.stack(jptc(x_adv))

for i in range(10):
    f, ax = plt.subplots(1,2)
    norm_orig_img = np.asarray(sample_data[i]['image'])
    perturbation = np.linalg.norm(norm_orig_img - x_adv[i])

    ax[0].set_title(f'Prediction: {classes[np.argmax(preds[i])]}')
    ax[0].imshow(np.array(sample_data[i]['image']).transpose(1,2,0))
    ax[0].set_xlabel('Original')


    ax[1].set_title(f'$\\ell ^{2}$={perturbation:.5f}\nPrediction adv: {classes[np.argmax(pred_adv[i])]}')
    ax[1].imshow(x_adv[i].transpose(1,2,0))
    ax[1].set_xlabel('Adversarial image')
    plt.show()
```

```bash
init adv example is not found, returning original image
init adv example is not found, returning original image
init adv example is not found, returning original image
init adv example is not found, returning original image
init adv example is not found, returning original image
init adv example is not found, returning original image
init adv example is not found, returning original image
init adv example is not found, returning original image
```

```{image} /_static/how-to/ic/bb-3.png
:alt: building
```

```{image} /_static/how-to/ic/bb-4.png
:alt: building
```

```{image} /_static/how-to/ic/bb-5.png
:alt: building
```

```{image} /_static/how-to/ic/bb-6.png
:alt: building
```

```{image} /_static/how-to/ic/bb-7.png
:alt: building
```

```{image} /_static/how-to/ic/bb-8.png
:alt: building
```

```{image} /_static/how-to/ic/bb-9.png
:alt: building
```

```{image} /_static/how-to/ic/bb-10.png
:alt: building
```

```{image} /_static/how-to/ic/bb-11.png
:alt: building
```

```{image} /_static/how-to/ic/bb-12.png
:alt: building
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
x_adv, y, metadata = attack(data=sample_data)

#Calc clean and robust accuracy
metric = AccuracyPerturbationMetric(jptc(sample_data), metadata)
metric.update(jptc(x_adv), y)
print(metric.compute())
```

```python
{'clean_accuracy': 0.8, 'robust_accuracy': 0.5, 'mean_delta': 28.528248}
```

```python
#Visualize the results
for img in x_adv[7:9]:
    plt.title(f'$\\ell ^{2}$={perturbation:.5f}\nPrediction adv: {classes[np.argmax(pred_adv[i])]}')
    plt.imshow(img.transpose(1,2,0))
    plt.show()
```

```{image} /_static/how-to/ic/bb-13.png
:alt: patch
```

```{image} /_static/how-to/ic/bb-14.png
:alt: patch
```

## 5. Targeted Black-Box Attack

 The following cells demonstrate execution of a targeted HopSkipJump attack against the model. Unlike an untargeted
 black-box attack, in this scenario an adversary is attempting to elicit a specific incorrect classification from the
 model. For instance, the following example demonstrates how an attack might trick a classifier to misclassify an
 engineering vehicle as a building. In order to achieve this, we simply provide an incorrect target label in the dataset
 we pass to the attack augmentation and set the targeted parameter to be True.

 It can sometimes be easier to successfully attack a model using a targeted attack as, with a well defined objective,
 the attack may converge quicker (and with few necessary iterations).

 Here we set the target label to be 0 in all instances of our sample data. An index of 0 corresponds to the class
 "Building".

 ```python
class TargetedImageDataset:
    
    def __init__(self, images):
        self.images = images
    def __len__(self)->int:
        return len(self.images)
    def __getitem__(self, ind: int):
        image = np.asarray(self.images[ind]['image'])
        y = np.zeros(len(classes))
        y[0] = 1
        return image, y, {}

targeted_sample_data = TargetedImageDataset(sample_data)
 ```

 The only other step is to add the targeted parameter and set it to True. This tells the HopSkipJump algorithm to
minimize the loss between the models prediction and the provided target label, in this case "Building".

```python
#defining and running attack
evasion_attack = HopSkipJump(
    classifier=jptc, verbose=True, targeted=True,
    init_size=100, max_eval=100, init_eval=100, max_iter=50,
)

attack = JaticAttack(evasion_attack)

x_adv, y, metadata = attack(data=targeted_sample_data, norm=2)

#visualization
preds = np.stack(jptc(sample_data))
pred_adv = np.stack(jptc(x_adv))

for i in range(8,10):
    f, ax = plt.subplots(1,2)
    norm_orig_img = np.asarray(sample_data[i]['image'])
    perturbation = np.linalg.norm(norm_orig_img - x_adv[i])

    ax[0].set_title(f'Prediction: {classes[np.argmax(preds[i])]}')
    ax[0].imshow(np.array(sample_data[i]['image']).transpose(1,2,0))
    ax[0].set_xlabel('Original')


    ax[1].set_title(f'$\\ell ^{2}$={perturbation:.5f}\nPrediction adv: {classes[np.argmax(pred_adv[i])]}')
    ax[1].imshow(x_adv[i].transpose(1,2,0))
    ax[1].set_xlabel('Adversarial image')
    plt.show()
```

```{image} /_static/how-to/ic/bb-15.png
:alt: patch
```

```{image} /_static/how-to/ic/bb-16.png
:alt: patch
```

## 5. Conclusion

We have successfully attacked a model with adversarial example that are based only on the model's in/output. In the next
steps, we can test several attacks at once on a model or we can attempt to defend the computed examples.

## 6. Next Steps

Check out other How-to Guides focusing on:

- [How to Simulate AutoAttack for Image Classification](auto_attacks.md)
- [How to Create Defenses for Image Classification](create_defenses.md)
- [How to Replace Datasets in Model Evaluation](replace_datasets.md)
