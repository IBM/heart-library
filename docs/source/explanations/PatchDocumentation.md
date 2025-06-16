# Overview - Creating Adversarial Patches

Adversarial patches important adversarial attacks due to their physical feasibility, and thus and essential part of any
testing suite and thus HEART. In this documentation, we provide the necessary context for generating such patches by
reviewing the necessary [background](#background), important design choices taken
[before crafting](#before-crafting-a-patch), [how to craft a patch](#crafting-the-patch),
[how to deal with difficulties when crafting a patch](#troubleshooting-crafting-patches) and how
[to evaluate a model against patches](#evaluating-models-on-patches). This documentation provides general background,
specifics of individual attacks can be found in the <project:../reference_materials/attack_cards/index.rst>.

## Background

Adversarial Patches are adversarial examples or evasion attacks. These attacks fool a computer vision model or an object
detector and make it output a wrong result [1,12]. For example, an object ‘airplane’ is recognized as a tree or not
detected at all. In mathematical terms, we add a perturbation {math}`\delta` to the input {math}`x`, such that the
model’s {math}`f`'s output disagrees with the original, correctly assigned output or class,

$$ f(x+\\delta) \\neq f(x) \\text{ [1,10].}$$

Figure 1. depicts an example from [1], where the authors depict adversarial examples for different classes, here the
numbers from one to two or three, respectively.

```{figure} ../_static/MNISTAttackExample.jpg
:scale: 50 %

Figure 1: MNIST example [1]. Along the y-axis (left) is the true class, and the x-axis (on top) denotes the 
classification output of a model.

```

$$\\min\_\\delta ||\\delta|| f(x+\\delta) \\neq f(x) \\text{ [1]}$$

This attack formulation is untargeted, as we only require a wrong output to succeed. In the formulation,
{math}`||\delta||` measures the change that {math}`\delta` adds to the sample. The reason to minimize this change is to
avoid the trivial solution of simply computing an image with an instance of another class. However, the exact
perturbation depends on the chosen norm. In most works [1,3], mathematical norms measure the overall change to all
pixels ({math}`L_\infty`), the squared distance ({math}`L_2`) or number of changed pixels ({math}`L_0`) [1]. While these
metrics integrate easily into optimization frameworks, they do not correspond to human perception [11,12]. In addition,
({math}`L_p`) norms may also not correspond to practically implementable changes [3,10]. Another, convenient solution is
thus, as suggested by Lian et al [5], to constrain the perturbation to a specific area of the image: to craft an
adversarial patch.

```{figure} ../_static/AdvAirplanes.jpg
:scale: 95 %

  Figure 2: Adversarial Patches on an object detector detecting planes [5]. The success of the patch is visible as the 
  purple bounding boxes around planes are missing. On the left, the patches (light green/blue) are on top of the 
  planes, on the right, the patches are next to them.

```

An example of this strategy is depicted in Figure 2. One may be surprised that such patches are malicious as they are
easy to spot. Yet, a human overseeing the system does not necessarily expect these patches to influence the system.
Below is an additional example from Kong et al. [2], where the patch is a billboard affecting an autonomous vehicle’s
AI. Kong et al. report that the poster affects the predicted steering angle, resulting in a wrong prediction being off
by up to 17°. Assuming an angle off by around 10°, observed after 15 frames, and that the vehicle has generous 3.3 foot
or 1 meter on its side, the vehicle needs 18.8 foot or 5.75 meter to leave its lane. Assuming a velocity of 30km/h or
12m/h, this distance is travelled in slightly more than 1 second.

```{figure} ../_static/AdvBillboard.jpg
:scale: 50 %

   Figure 3: Adversarial Billboard [2]. At the top left, the original billboard is depicted, at the top right the 
   adversarial counterpart. In the bottom, the billboard is depicted next to a road, as visible to vehicle and driver.

```

In this library, we focus on patches and explain how to use HEART to craft a patch and outline relevant considerations
on how to evaluate the success or impact of a patch attack on a model.

## Before Crafting a Patch

If the patch is crafted as part of a model evaluation, is important to first understand the exact threat model we
evaluate [3,10]. This translates to understanding what knowledge the attacker potentially has about the model and
choosing the attack’s goal, and how this goal is evaluated. For the sake of patch attacks, where the kind of change
induced in the sample is set, there are two main choices regarding what knowledge the attacker has about the model:
[white-box or black-box attacks](/explanations/white_vs_black_box)

So far, we have mainly defined white-box and black-box in relation to the model. The same considerations could be made
for the data. However, this is primarily relevant in cases where, for example, the representations of the data are
subject to design like in Malware detection [10], and less relevant in the vision tasks discussed here [10].

Another relevant decision concerns the attack type and goal. The attack goal describes what the attacker aims to achieve
with the perturbation in terms of desired behavior of the classifier. These goals can be very task specific, and
partially differ across object detection and image classification, several goals exist [3]. For more information on
attack types and their respective goals, see [this explanation](/explanations/attack_types).

We will now address the question on how to [evaluate](/explanations/evaluation_metrics). these goals numerically. First,
however, we have to acknowledge that benign accuracy (e.g., on clean, unperturbed data) is also relevant to security and
should not be missed in any (security) evaluation [10]. As reasoned by Biggio and Roli [10], ideally, the benign model's
accuracy is plotted against attack performance measured by the introduced perturbation to an image. This perturbation,
as discussed in the [background](#background) is measured using {math}`L_p` norms. The accuracy must be measured
according to the desired goal. For untargeted attacks, the deviation from the ground truth is computed; for targeted
attacks, we compute the match with the attack's target class. In the case of invisibility, we have to compare to a
ground truth with the target object removed, and denial of service by for example counting the number of outputted
objects. Specific attack goals and evaluation can be found in the
<project:../reference_materials/attack_cards/index.rst>.

## Crafting the Patch

With the previous choices set, we can start crafting the patch. The access to the model affects the overall experimental
setting and the attack. For example, the
[GRAPHITE](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#graphite-blackbox)
attack from ART can be applied in black-box settings, but the same attack can also be applied to white-box models. The
attacker’s goal is specified when instantiating the attack (for example via an argument, in case of GRAPHITE called y,
which sets the target class). Having chosen the attack and decided on a possible target, we can run the attack with
default parameters.

At this point, it helps to understand that crafting a patch is a learning procedure, as is training the network itself.
However, when crafting a patch, the patch is updated and the model’s weights remain fixed. Hence, any attack method has
parameters that are expected in any learning or optimization procedure, like learning rate, iterations, batch size, etc.
Analogous to any other learning problem, the loss can be monitored during the learning procedure to understand if the
attack is converging well.

As training a neural network, training a patch can be tedious, and the patch may not converge. In theory, an attack
should always be able to reduce the model's accuracy to zero with arbitrary attack strength (e.g., perturbation size,
optimization length). Attacks may however fail for various reasons, including that the model does not provide enough
information [9]. In these cases, or generally if the attack does not perform well when tested on hold-out data, it is
advised to increases the samples the attack is optimized on, to increase optimization time, or to add scaling and
rotation operations available to implicitly increase the data and avoid overfitting.

Optimizing attacks, as stated above can be tedious, but best practices should still be respected. This related to a
clear separation of data used to optimize and test the patch, and to not add perturbation outside the normal range of
pixel values (clipping), for example. This is especially relevant in a testing setting, as the results of the evaluation
should be faithful and thus represent the attacker's intended impacts. To test the here provided knowledge, we would
like to point the interested reader to
[Notebook 4](https://gitlab.jatic.net/jatic/ibm/hardened-extension-adversarial-robustness-toolbox/-/blob/main/notebooks/4_get_started_adversarial_patch.ipynb?ref_type=heads)
about patches for vision models and
[Notebook 6](https://gitlab.jatic.net/jatic/ibm/hardened-extension-adversarial-robustness-toolbox/-/blob/4faa6972b16cb28ef75177f3355472c28b1ce253/notebooks/6_adversarial_patch_for_object_detection.ipynb)
about patches for object detection.

## Troubleshooting Crafting Patches

As reasoned above, patches may be tedious to optimize, and they may fail to converge entirely. In this section, we
briefly review which steps can be taken to find a core problem. While this list has a rough ordering, the error you face
may not be included in this list.

1. Verify all inputs / outputs of both model and attack.
1. Check that the gradients the attacks computes are non-zero, e.g., the attack has information to work on.
1. Control that the attack's loss decreases on several individual samples (one each) with unconstrained perturbation.
   These samples should (eventually be misclassified).

Applying these suggestions, monitoring and repeating this procedure for several samples and eventually batches should
result in a functional patch.

## Evaluating Models on Patches

After having seen the success of the patch on the model, one may be inclined to harden the model. However, abundant
research from the adversarial example literature suggests that defending adversarial examples is hard [6-8]. To give an
example, Athalye et al. [8] showed that often, attacks do not perform worse against defended models because the models
are more robust, but instead because the attacks fail on the hardened models as, for example, the gradients are not
providing enough information for the attack to find a good perturbation {math}`\delta`. Should a defense be developed,
we suggest reading the document by Carlini et al. [6] covering the list of common flaws in defense evaluation.

We recap the here on a high level to outline possible problems in an evaluation.

- **Train and test split** Keep train/test sets separate when evaluating defenses. As a best practice in general model
  training, this avoids overfitting and thus a wrong assessment of the capabilities of your defense.
- **Adaptive attacks** Test an adaptive attack, e.g. an attack that exploits the defense’s mechanics. As an example,
  consider a defense that rejects all outputs if they are {math}`<0.8`. An adaptive attack has then a special condition
  that the output of the classifier should be {math}`>0.8` for the desired perturbation.
- **Reproducibility** Release code and models to ease security analysis for others. Similarly, rely on implementations
  such as ART or HEART, or secondly the author's code, to avoid a wrong assessment by implementation errors.
- **Known Baselines** Use established models for which attack performances are well known. For these models, check if
  observed numbers align with previous findings. In the case that your results differ greatly from previous results, try
  to determine the source of the difference.
- **Eventual Attack success** When running an attack, parameters like the perturbation size, but also the number of
  iterations determine the attack strength. With arbitrary attack strength, the attack should eventually succeed in
  fooling the model.
- **Attack diversity** A single attack may fail and thus yield a wrong assessment. Ideally, models are tested with a
  variety of different attacks, including white-box and black-box attacks to increase diversity.
- **Transfer attacks** Train a second, undefended model on the same data and test an attack crafted on that model. This
  fixes issues like gradient-masking, where the defended model is still vulnerable, but the attack fails to find a valid
  perturbation.
- **Randomness** Randomness is often a secret within a defense, but thus yields security-by-obscurity. If possible, it
  should be incorporated into an evaluation. An example is to repeatedly query the model to remove randomness added to
  the outputs.
- **Simple baselines** It may be easier than expected to fool a model. It is thus good practice to also compare to
  performance when masking, blurring, or otherwise changing the area of the patch. This yields a baseline on how
  important the features under the patch are in general.
- **Ablation studies** Perform ablation studies of all attack and defense hyperparameters like perturbation strength,
  thresholds, etc. This avoids overlooking settings which are performing better than the current setting, or
  inconsistent behavior of the defense when varying hyperparameters.
- **Attack failure** Control for attack failure modes [9]. This includes, for example, gradients with insufficient
  information to derive a perturbation (gradient masking), but also entails problems as simple as bugs that are in the
  code used.

While this list provides many useful starting points to find flawed model evaluations, it is not complete and may not
contain the underlying problem in your specific case.

## References

[1] Papernot, Nicolas, et al. The limitations of deep learning in adversarial settings. 2016 IEEE European symposium on
security and privacy (EuroS&P). IEEE, 2016.

[2] Kong, Zelun, et al. Physgan: Generating physical-world-resilient adversarial examples for autonomous driving.
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[3] Grosse, Kathrin, and Alexandre Alahi. A qualitative AI security risk assessment of autonomous vehicles.
Transportation Research Part C: Emerging Technologies 169 (2024): 104797.

[4] Gao, Lianli, et al. Patch-wise attack for fooling deep neural network. Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16. Springer International Publishing, 2020.

[5] Lian, Jiawei, et al. Benchmarking adversarial patch against aerial detection. IEEE Transactions on Geoscience and
Remote Sensing 60 (2022): 1-16.

[6] Carlini, Nicholas, et al. On evaluating adversarial robustness. arXiv preprint arXiv:1902.06705 (2019).

[7] Tramer, Florian, et al. On adaptive attacks to adversarial example defenses. Advances in neural information
processing systems 33 (2020): 1633-1645.

[8] Athalye, Anish, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security:
Circumventing defenses to adversarial examples. International conference on machine learning. PMLR, 2018.

[9] Pintor, Maura, et al. Indicators of attack failure: Debugging and improving optimization of adversarial examples.
Advances in Neural Information Processing Systems 35 (2022): 23063-23076.

[10] Biggio, Battista, and Fabio Roli. Wild patterns: Ten years after the rise of adversarial machine learning.
Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security. 2018.

[11] Zhao, Zhengyu, Zhuoran Liu, and Martha Larson. Towards large yet imperceptible adversarial image perturbations with
perceptual color distance. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[12] Sharif, Mahmood, Lujo Bauer, and Michael K. Reiter. On the suitability of lp-norms for creating and preventing
adversarial examples. Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2018.

[13] Hu, Shengnan, et al. "Cca: Exploring the possibility of contextual camouflage attack on object detection." 2020
25th International Conference on Pattern Recognition (ICPR). IEEE, 2021.

[14] Liu, Chengji, et al. "Object detection based on YOLO network." 2018 IEEE 4th information technology and
mechatronics engineering conference (ITOEC). IEEE, 2018.
