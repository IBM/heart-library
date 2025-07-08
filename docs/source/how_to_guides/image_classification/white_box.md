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
Test & Evaluation of a small benchmark dataset based on drone imagery. The first attack we will use is Projected
Gradient Descent (PGD), a simple attack based on the model gradients warranted the input. We then visualize the attack
and afterwards constrain the change applied to the image to a fraction of the input image using a patch attack.
Computing the performance under these white-box attacks is a crucial step in T&E.

:::: {grid} 5
:::{grid-item} **Intended Audience:** All T&E Users
:::
:::{grid-item} **Requirements:** Basic Python and Torchvision / ML Skills
:::
:::{grid-item} **Notebook Runtime:** Full run of the notebook: <10 minutes
:::
:::{grid-item} **Reading time:** ~10 Minutes
:::
:::{grid-item} **Order of Completion:** 1-2.; 3. and 4. can be done in any order or independently.
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

```{note}
The [How to Simulate White-Box Attacks for Image Classification Companion Notebook](https://github.com/IBM/heart-library/blob/main/notebooks/how_tos/image_classification/1_How_to_Simulate_White-box_attacks_for_Image_Classification.ipynb)
can be downloaded via the HEART public GitHub.
```

:::

::::

### Contents

1. Imports and set-up
1. Load data and model
1. Projected Gradient Descent Attack
1. Patch attack
1. Targeted White-box Attack
1. Conclusion
1. Next Steps

### Learning Objectives

- How to define a custom model and use drone imagery
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

data = load_dataset("CDAO/xview-subset-classification", split="test[0:10]")
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

```text
 Resolving data files:   0%|          | 0/31 [00:00<?, ?it/s]
```

```{image} /_static/how-to/ic/wb-1.png
:alt: Building
```

We then load a custom model which comes with the repository. Most important is that the model has the correct input
shape and is trained to perform decently well on the data. At the bottom, we wrap the model into a
JaticPyTorchClassifier.

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
plt.imshow(data[0]['image'])
_ = plt.title(f'benign classification: {classes[np.argmax(np.stack(pred_batch[0]))]}')
plt.show()
```

```{image} /_static/how-to/ic/wb-2.png
:alt: Building
```

## 3. Define and Run PGD Attack

We are now ready to define the first attack, Projected Gradient Descent (PGD). We then apply the attack to the first
image in our test data, applying many iterations to ensure that the images fool the classifier. We first plot the result
and then compute the benign and adversarial performance on a larger batch. From there we observe that while the initial
performance is correct, the final classification is wrong (however we are computing the metric only on one sample).

When rerunning this notebook, this part can be counted as completed if a correctly classified sample is misclassified.

```python
#define attack
pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=100, eps=0.1, eps_step=0.1, targeted=False)

#wrap and run attack
jattack = JaticAttack(pgd_attack, norm=2)
x_adv, y, metadata = jattack(data=sample_data)

#plot adversarial counterpart of image above
pred_batch = jptc(x_adv)
plt.imshow(x_adv[1].transpose(1,2,0))
_ = plt.title(f'adversarial classification: {classes[np.argmax(pred_batch,axis=1)[1]]}')
plt.show()
```

```text
PGD - Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

```{image} /_static/how-to/ic/wb-3.png
:alt: oil tanker
```

```python
#using the adversarial examples from above, computing performance metric
groundtruth_target_batch = data['label'][0:10]
benign_preds_batch = jptc(sample_data)
adversarial_preds_batch = jptc(x_adv)

metric = AccuracyPerturbationMetric(benign_preds_batch, metadata)
metric.update(adversarial_preds_batch, groundtruth_target_batch)
print(metric.compute())
```

```text
 {'clean_accuracy': 0.8, 'robust_accuracy': 0.0, 'mean_delta': 33.070686}
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
learning_rate = 0.001 #500
max_iter = 700
patch_shape = (3, 45, 45)
patch_location = (0,0)

ap = JaticAttack(AdversarialPatchPyTorch(estimator=jptc, rotation_max=rotation_max, patch_location=patch_location,
                      scale_min=scale_min, scale_max=scale_max, patch_type='circle',
                      learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size,
                      patch_shape=patch_shape, verbose=True, targeted=False))

patched_images, _, metadata = ap(data=sample_data)
patch = metadata[0]["patch"]
patch_mask = metadata[0]["mask"]

#plot patch
plt.axis("off")
plt.imshow(((patch) * patch_mask).transpose(1,2,0))
_ = plt.title('Generated Adversarial Patch')
plt.show()
```

```text
Adversarial Patch PyTorch:   0%|          | 0/700 [00:00<?, ?it/s]
```

```{image} /_static/how-to/ic/wb-4.png
:alt: patch
```

```python
 #plot the images with the patch applied and prediction
preds = jptc(patched_images)
for i, patched_image in enumerate(patched_images[0:1]):
    _ = plt.title(f'''Prediction: {classes[np.argmax(preds[i])]}''')
    plt.imshow(patched_image.transpose(1,2,0))
    plt.show()
for i, patched_image in enumerate(patched_images[7:9]):
    _ = plt.title(f'''Prediction: {classes[np.argmax(preds[i])]}''')
    plt.imshow(patched_image.transpose(1,2,0))
    plt.show()
```

:::: {grid} 3
:::{grid-item}

```{image} /_static/how-to/ic/wb-5.png
:alt: building
```

:::
:::{grid-item}

```{image} /_static/how-to/ic/wb-6.png
:alt: building
```

:::
:::{grid-item}

```{image} /_static/how-to/ic/wb-7.png
:alt: building
```

:::
::::

While we see that the patch does not work for images of the class 'building', it turns the outputs of other classes'
images into buildings.

## 5. Targeted White-box Attack

The following cells demonstrate execution of a targeted PGD attack against the model. Unlike an untargeted attack, in
this scenario and adversary is attempting to elicit a specific incorrect classification from the model. For instance,
the following examples demonstrates how an attack might trick a classifier to misclassify a building as an oil tanker.
In order to achieve this, we simply provide an incorrect target label in the dataset we pass to the attack augmentation
and set the targeted parameter to be True.

It can sometimes be easier to successfully attack a model using a targeted attack as, with a well defined objective, the
attack may converge quicker (and with few necessary iterations).

Here we set the target label to be 4 in all instances of our sample data. An index of 4 corresponds to the class "Oil
Tanker".

```python
class TargetedImageDataset:
    
    def __init__(self, images):
        self.images = images
    def __len__(self)->int:
        return len(self.images)
    def __getitem__(self, ind: int):
        image = np.asarray(self.images[ind]['image'])
        return image, 4, {}

targeted_sample_data = TargetedImageDataset(sample_data)
```

The only other step is to add the targeted parameter and set it to True. This tells the PGD algorithm to minimize the
loss between the models prediction and the provided target label, in this case "Oil Tanker".

```python
#define attack
pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=100, eps=0.1, eps_step=0.1, targeted=True)

#wrap and run attack
jattack = JaticAttack(pgd_attack, norm=2)
x_adv, y, metadata = jattack(data=targeted_sample_data)

#plot adversarial counterpart of image above
pred_batch = jptc(x_adv)
plt.imshow(x_adv[1].transpose(1,2,0))
_ = plt.title(f'adversarial classification: {classes[np.argmax(pred_batch,axis=1)[1]]}')
plt.show()
```

```text
PGD - Batches:   0%|          | 0/1 [00:00<?, ?it/s]
```

```{image} /_static/how-to/ic/wb-8.png
:alt: building
```

## 6. Conclusion

We have successfully attacked a model with adversarial example that are based on the model's gradients. In the next
steps, we can attempt to defend the computed examples or decrease the knowledge the attacker has by running black box
attacks which are not based on the model's gradients.

## 7. Next Steps

Check out other How-to Guides focusing on:

- [How to Simulate Black-box Attacks for Image Classification](black_box.md)
- [How to Simulate AutoAttack for Image Classification](auto_attacks.md)
- [How to Create Defenses for Image Classification](create_defenses.md)
- [How to Replace Datasets in Model Evaluation](replace_datasets.md)
