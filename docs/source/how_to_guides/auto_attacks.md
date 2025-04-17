How to Simulate Auto Attacks
============

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
This notebook provides a beginner friendly introduction to using auto attack on image classification as part of Test & Evaluation of a small benchmark dataset (CIFAR-10). Auto attack is an ensemble or a collection of other attacks or attack configuration, which can be run in parallel. In this notebook, we learn how to instantiate the attack and deduce which attack from all is the most successful. Testing an ensemble of attacks for best performance is a crucial step in T&E.

:::: {grid} 4
:::{grid-item}
**Intended Audience:** All T&E Users
:::
:::{grid-item}
**Requirements:** Basic Python and Torchvision / ML Skills
:::
:::{grid-item}
**Notebook Runtime:** ~2 Minutes
:::
:::{grid-item}
**Reading time:** ~10 Minutes
:::
::::

::::{grid} 2

:::{grid-item}
:columns: 8
Before you begin, you will want to make sure that you download the how-to guide's companion Jupyter notebook.  This notebook allows you to follow along in your own environment and interact with the code as you learn.  The code snippets are also included in the documentation, but the notebook is provided for ease of use and to enable you to try things on your own.
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
2. Load data and model
3. Auto-Attack Initialization
4. Auto-Attack Evaluation
5. Calculate Clean and Robust Accuracy
6. Further Evaluation: Plot Samples, Best Attacks
7. Conclusion
8. Next Steps

### Learning Objectives
- How to define a custom model and use CIFAR 10
- How to run an auto attack with HEART

## 1. Imports and Set-up
We import all necessary libraries for this tutorial. In this order, we first import general libraries such as numpy, then load relevant methods from ART. We then load the corresponding HEART functionality and specific torch functions to support the model. Lastly, we use a command to plot within the notebook.

```python
import numpy as np
import os
import torch
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

# ART imports
from art.utils import load_dataset
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.attacks.evasion.auto_attack import AutoAttack

# HEART imports
from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
from heart_library.attacks.attack import JaticAttack
from heart_library.metrics import AccuracyPerturbationMetric

# MAITE import for evaluation
from maite.protocols.image_classification import Dataset as ic_dataset

# using matplotlib inline to see the figures
%matplotlib inline
```

## 2. Load CIFAR-10 Data and Model for Classification
We now load the data, importing only a small part to save compute for this small demonstration. We then define the model and wrap it as JATIC pytorch classifier.

Here, we first define the CIFAR10 labels and  then use ART's utility method for easily loading CIFAR-10 images as numpy arrays. We use a subset of the data to save runtime, a total of 25 samples. In addition, we load a dataset as a modified dataframe to be compatible with JATIC.

```python
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
i = 25
x_train = x_train[:i, :].transpose(0, 3, 1, 2).astype('float32')
x_test = x_test[:i, :].transpose(0, 3, 1, 2).astype('float32')
y_train = y_train[:i, :].astype('float32')
y_test = y_test[:i, :].astype('float32')

class ImageDataset:
    
    metadata = {"id": "example"}
    
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self)->int:
        return len(self.images)
    def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        image = self.images[ind]
        label = self.labels[ind]
        return image, label, {}
    
data = ImageDataset(x_train, y_train)

assert isinstance(data, ic_dataset)
```

We load a simple classification model from the repo and wrap it in JaticPyTorchClassifier so the model is compatible with  MAITE .

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
        # x = torch.nn.functional.softmax(x, dim=1) # removed to return logits
        return x

# Define the network
model = Model()

# Define a loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Get classifier
ptc = JaticPyTorchClassifier(
    model=model, loss=loss_fn, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=(10), clip_values=(0, 1), channels_first=False,
)

#plot original image
pred_batch = ptc(x_train[[0]])
plt.imshow(x_train[0].transpose(1,2,0))
_ = plt.title(f'benign classification: {labels[np.argmax(np.stack(pred_batch))]}')
plt.show()
```

```{image} /_static/how-to/aa-1.png
:alt: Frog
```

The above picture shows, in a resolution of 32 x 32 pixels, the image of a frog with the model's classification output above.

## 3. Auto-Attack Initialization

We now initialize AutoAttack. To this end, we create a number of differently initialized attacks to carry out the T&E. For example, we:
- vary the max_iter values, but keep all other parameters fixed,
- and vary the eps_step values, but keep all other parameters fixed.

These values serve as a mere example, other parameters can/should be further explored, as well as applying different attacks. 

**Important I:** If we use attacks that are untargeted and ask auto attack to executed targeted mode, the attacks will still be executed, but in untargeted mode. **Important II:** We have to set the parallel pool size to to enable parallel AutoAttack:

- Setting a multiprocess pool size value allows AutoAttack to execute across a specified number of worker processes.
- By default, the AutoAttack parallel_pool_size value is 0, which disables the usage of parallel AutoAttack.

In case there is no intuition about which parameters could be tested, we show below a default initialization for AutoAttack's parameters.

```python
n = 20 # number of iterations
attacks = []

# create multiple values for max_iter
max_iter = [i+1 for i in list(range(n))]
print('max_iter values:', max_iter)

# create multiple values for eps
eps_steps = [round(float(i/100+0.001), 3) for i in list(range(n))]
print('eps_step values:', eps_steps)

# make a list with all attack/parameter combinations we want to test
for i, eps_step in enumerate(eps_steps):
    attacks.append(
        ProjectedGradientDescentPyTorch(
            estimator=ptc,
            norm=np.inf,
            eps=0.1,
            eps_step=eps_step,
            max_iter=10,
            targeted=False,
            batch_size=32,
            verbose=False
        )
    )
    attacks.append(
        ProjectedGradientDescentPyTorch(
            estimator=ptc,
            norm=np.inf,
            eps=0.1,
            max_iter=max_iter[i],
            targeted=False,
            batch_size=32,
            verbose=False
        )
    )
    
print('Number of attacks:', len(attacks))

# add the attacks to Autoattack to manage execution, and wrap with JAticattack to support MAITE
jatic_attack_notparallel = JaticAttack(AutoAttack(estimator=ptc, attacks=attacks, targeted=True), norm=2)

# Now figure out parallel usage.
# You can modify this value to either raise or lower the number of multiprocess workers spawned by Parallel AutoAttack
parallel_pool_size = 2

print(f"Count of total (logical) CPUs available: os.cpu_count() = {os.cpu_count()}")
print(f"Pool size to use during parallel AutoAttack: parallel_pool_size = {parallel_pool_size}")

jatic_attack_parallel = JaticAttack(AutoAttack(estimator=ptc, attacks=attacks, targeted=True, parallel_pool_size=parallel_pool_size), norm=2)
# for comparison: autoattack purely based on ART
core_attack = AutoAttack(estimator=ptc, attacks=attacks, targeted=True)




## In case you have no intuition about possible parameters, defaults can be used:
#attack = JaticAttack(AutoAttack(
#    estimator=ptc,
#    targeted=True,
#    parallel=True,
#))
```

```python
max_iter values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
eps_step values: [0.001, 0.011, 0.021, 0.031, 0.041, 0.051, 0.061, 0.071, 0.081, 0.091, 0.101, 0.111, 0.121, 0.131, 0.141, 0.151, 0.161, 0.171, 0.181, 0.191]
```

```python
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[4], line 16
     12 # make a list with all attack/parameter combinations we want to test
     13 for i, eps_step in enumerate(eps_steps):
     14     attacks.append(
     15         ProjectedGradientDescentPyTorch(
---> 16             estimator=ptc,
     17             norm=np.inf,
     18             eps=0.1,
     19             eps_step=eps_step,
     20             max_iter=10,
     21             targeted=False,
     22             batch_size=32,
     23             verbose=False
     24         )
     25     )
     26     attacks.append(
     27         ProjectedGradientDescentPyTorch(
     28             estimator=ptc,
   (...)
     35         )
     36     )
     38 print('Number of attacks:', len(attacks))

NameError: name 'ptc' is not defined
```

## 4. Auto-Attack Evaluation
We are now executing the attacks using ART core, HEART in non-parallel mode, and HEART in parallel mode. For each of these cases, we compute whether the attack is robust: if it is, predictions for each image will be incorrect.

When rerunning this notebook with your own parameters, all the attacks (or your attack) should be fully robust.

```python
# Run core ART attack
core_adv = core_attack.generate(x=x_train, y=y_train)
predictions_core_adv = np.sum(np.argmax(np.stack(ptc(core_adv)), axis=1) == np.argmax(y_train, axis=1)) == 0
print(f'Is core ART attack fully robust: {predictions_core_adv}')

# Run HEART attack in non-parallel mode
nonparallel_adv, y_nonparallel, metadata_nonparallel = jatic_attack_notparallel(data=data)
predictions_jatic_notparallel = np.sum(np.argmax(np.stack(ptc(nonparallel_adv)), axis=1) == np.argmax(y_train, axis=1)) == 0
print(f'Is HEART non-parallel attack fully robust: {predictions_jatic_notparallel}')

# Run HEART attack in parallel mode
parallel_adv, y_parallel, metadata_parallel = jatic_attack_parallel(data=data)
predictions_jatic_parallel = np.sum(np.argmax(np.stack(ptc(parallel_adv)), axis=1) == np.argmax(y_train, axis=1)) == 0
print(f'Is HEART parallel attack fully robust: {predictions_jatic_parallel}')
```

```python
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[5], line 2
      1 # Run core ART attack
----> 2 core_adv = core_attack.generate(x=x_train, y=y_train)
      3 predictions_core_adv = np.sum(np.argmax(np.stack(ptc(core_adv)), axis=1) == np.argmax(y_train, axis=1)) == 0
      4 print(f'Is core ART attack fully robust: {predictions_core_adv}')

NameError: name 'core_attack' is not defined

```

## 5. Calculate Clean and Robust Accuracy

In addition to the above robustness, we compute the three different numbers: the clean accuracy, the robust accuracy, and the perturbation size. We define:

- Clean accuracy: the accuracy of the classification model on the original, benign, images
- Robust accuracy: the accuracy of the classification model on the adversarial images
- Perturbation size: this is the average perturbation added to all images which successfully fooled the classification model

```python
 benign_pred_batch = ptc(x_train)
groundtruth_target_batch = np.argmax(y_parallel, axis=1)
adv_pred_batch = ptc(parallel_adv)

metric = AccuracyPerturbationMetric(np.stack(benign_pred_batch), metadata_parallel)
metric.update(np.stack(adv_pred_batch), groundtruth_target_batch)
print(metric.compute())
```

```python
{'clean_accuracy': 0.56, 'robust_accuracy': 0.0, 'mean_delta': 1.8226798}
```
## 6. Further Evaluation: Visualize Samples and Crafted Examples, Best Attacks

We plot the data in four columns, including the original sample, and the three examples generated by AutoAttack in the order: ART core AutoAttack, ART JATIC AutoAttack non-paralell, and ART JATIC in parallel mode. 

- The first collum with the original image shows the ground truth label and the classifier prediction on the original image. Note that as we are passing ground truth labels to the generate method, attacks will not add perturbations to images that are already misclassified. Perturbations are only added to images in which the classifier made correct predictions.
- The second column images are adversarial images generated by ART core AutoAttack. L2 distance between the original and adversarial image is shown. 
- The third column images are adversarial images generated by ART JATIC AutoAttack in non-parallel mode - these should be identical to ART core (a sanity check).
- The fourth column images are adversarial images generated by ART JATIC AutoAttack in parallel mode. Note how parallel model achieves lower L2 distance between the original and adversarial image - as all jobs are run in parallel mode, more attacks can be evaluated and successful attacks with lower perturbations added can be selected.

```python
# to plot all images
#for i in range(len(x_train)):
for i in range(4):
    f, ax = plt.subplots(1,4, constrained_layout = False)

    core_perturbation = np.linalg.norm(x_train[i] - core_adv[[i]])
    nonparallel_perturbation = np.linalg.norm(x_train[i] - nonparallel_adv[i])
    parallel_perturbation = np.linalg.norm(x_train[i] - parallel_adv[i])
    
    pred_core_benign = ptc.predict(x_train[i])
    pred_core_adv = ptc.predict(core_adv[i])
    pred_non_parallel_adv = np.stack(ptc(nonparallel_adv[i]))
    pred_parallel_adv = np.stack(ptc(parallel_adv[i]))

    ax[0].set_title(f'GT: {labels[np.argmax(y_train[i])]};  Pred: {labels[np.argmax(pred_core_benign)]}')
    ax[0].imshow(x_train[i].transpose(1,2,0))
    ax[0].set_xlabel('Original')
    
    ax[1].set_title(f'{labels[np.argmax(pred_core_adv)]} ($\\ell ^{2}$={core_perturbation:.5f})')
    ax[1].imshow(x_train[i].transpose(1,2,0))
    ax[1].set_xlabel('Core')

    ax[2].set_title(f'{labels[np.argmax(pred_non_parallel_adv)]} ($\\ell ^{2}$={nonparallel_perturbation:.5f})')
    ax[2].imshow(x_train[i].transpose(1,2,0))
    ax[2].set_xlabel('Non-parallel')

    ax[3].set_title(f'{labels[np.argmax(pred_parallel_adv)]} ($\\ell ^{2}$={parallel_perturbation:.5f})')
    ax[3].imshow(x_train[i].transpose(1,2,0))
    ax[3].set_xlabel('Parallel')
    f.set_figwidth(15)
    plt.show()
```

```{image} /_static/how-to/aa-2.png
:alt: truck
```
```{image} /_static/how-to/aa-3.png
:alt: truck
```
```{image} /_static/how-to/aa-4.png
:alt: truck
```
```{image} /_static/how-to/aa-5.png
:alt: truck
```
AutoAttack has the additional feature, that it will, for any input sample, return the best performing attack and its configuration. We will plot this now for inspection.

```python
print(jatic_attack_parallel._attack)
```

```python
AutoAttack(targeted=True, parallel_pool_size=2, num_attacks=400)
BestAttacks:
image 1: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.001, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10, random_eps=False, verbose=False, )
image 2: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 3: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.011, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10, random_eps=False, verbose=False, )
image 4: n/a
image 5: n/a
image 6: n/a
image 7: n/a
image 8: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 9: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.001, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=10, random_eps=False, verbose=False, )
image 10: n/a
image 11: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 12: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 13: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 14: n/a
image 15: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 16: n/a
image 17: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 18: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 19: n/a
image 20: n/a
image 21: n/a
image 22: n/a
image 23: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 24: ProjectedGradientDescentPyTorch(norm=inf, eps=0.1, eps_step=0.1, targeted=True, num_random_init=0, batch_size=32, minimal=False, summary_writer=None, decay=None, max_iter=2, random_eps=False, verbose=False, )
image 25: n/a
```


## 7. Conclusion

We have learned that AutoAttack applies a collection of attack, how to initialize this collection, and how to determine which performed best. AutoAttack is an easy way to security test an existing model, as it runs different parameters and attacks without us explicitly running and comparing them.

## 8. Next Steps

Check out other How-to Guides focusing on:

- [Black-Box Attacks](black_box.md)
- [White-Box Attacks](white_box.md)