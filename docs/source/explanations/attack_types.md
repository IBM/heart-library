# Attack Types

AI is vulnerable to evasion, poisoning, inference, and extraction attacks. **HEART currently focuses on the evaluation
of computer vision models – including image classification and object detection – against evasion attacks**, but will
soon be extended to include the other three types of threat.

## Evasion

Attacks that involve the manipulation of runtime data input into a trained model and crafting perturbations that
cause the model to perform poorly (sometimes in a specific, targeted way). [^1]

HEART supports the following types of evasion attacks:

- Hop skip jump
- Laser
- Query efficient Black Box
- [Projected Gradient Descent](/reference_materials/attack_cards/projected_gradient_descent)
- [Patch](/reference_materials/attack_cards/patch_attack)

Evasion attacks may have different goals.

For object detection models, evasion attacks may aim to cause the model to hallucinate objects (_appearing attack_),
ignore existing objects (_hiding attack_ or _invisibility_) [^2], or _mislocate_ or _misclassify_ detected objects.
Variations of the attack goal for object detection can include overflow attacks that lead the model to detect an
extremely large number of objects. This can result in memory overflows and/or increased latency [^3], rendering the
detector unusable (_denial of service_).

Evasion attacks can be either _untargeted_ (the output is any valid output other than the correct output) or _targeted_
(the output is incorrect and has been defined as a specific target by the attacks). In the case of image classification,
if the correct classification is ‘airplane’, any classification other than ‘airplane’ caused by the crafted adversarial
perturbation is a successful _untargeted_ attack. In most cases, untargeted attacks shift the adversarial point to any
close neighboring class. On the other hand, _targeted_ attacks aim to cause the classifier to output a specific target
class. Adversarial perturbations for targeted attacks are often harder to craft than for untargeted attacks, as they
need to find a perturbation that shifts the point into a specific class, not just any neighboring class.

## Poisoning

Poisoning means that the attacker introduces specifically crafted, _poisonous_ samples into the training data with
the goal of inserting a _backdoor_ into the trained model to influence its behavior at runtime, make it more difficult
to train the model on that data, or change its behavior in unexpected ways like confusing two image categories. [^4]

## Inference

Inference attacks, also know as privacy attacks, do not directly affect model performance. Instead, these attacks infer
private or sensitive information in the training dataset of the model by interacting with that model (without any access
to the training data itself). Attackers probe the model with specifically selected input samples and analyze model
output to derive the targeted insights.

## Extraction

In an extraction attack, the attacker will attempt to decipher information about the model.  This including its
architecture and parameters, under special circumstances, in order to replicate its functionality and steal its
intellectual property.

Attacks can be staged in concert to increase their combined effectiveness. For example, extraction attacks can provide
key information to enable stronger [white-box](white_vs_black_box) evasion attacks, or to extract a model that could
later leak private information in a strong white-box inference attack.

## References

<!-- References -->
[^1]: Muthalagu, Raja, Jasmita Malik and Pranav M Pawar. "Detection and prevention of evasion attacks on machine learning
models." Expert Systems with Applications 266 (25 March 2025): 126044.

[^2]: Hu, Shengnan, et al. "Cca: Exploring the possibility of contextual camouflage attack on object detection." 2020 25th
International Conference on Pattern Recognition (ICPR). IEEE, 2021.

[^3]: Shapira, Avishag, et al. "Phantom sponges: Exploiting non-maximum suppression to attack deep object detectors."
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.

[^4]: Koffas, Stefanos, Jing Xu, Mauro Conti and Stjepan Picek. "Can You Hear It? Backdoor Attacks via Ultrasonic
Triggers." 10.48550/arXiv.2107.14569. 2021.
