# White- vs. Black-Box Attacks

Based on the amount of information that the attacker has, attacks can be classified as either black-box or white-box.

In a **black-box attack**, the attacker does not have any information or access to the model, but can submit inputs to
the model and observe its output.

In black-box evasion attacks, gradients of the model’s loss surface can be approximated based on sequences of
specifically selected inputs and used to craft adversarial examples with minimal adversarial perturbations. The critical
performance metric is the number of queries that must be made to the model to minimize the adversarial perturbations
[1,2].

In contrast, **white-box attacks** require detailed knowledge of and access to the model including its architecture,
parameters, and the ability to calculate loss gradients by numerical differentiation or approximation [3].

These attacks are the strongest possible attacks and represent the worst-case scenario for the defender of the model.
Calculated loss gradients allow the attacker to optimize the creation of minimal perturbations that are specific to the
model, allowing for an attack which is less perceptible and more effective. Nonetheless, white-box attacks can still
fail despite the attacker’s knowledge of the model [10].

## References

[1] Chen, Shang-Tse, et al. Shapeshifter: Robust physical adversarial attack on faster r-cnn object detector. Machine
Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2018, Dublin, Ireland, September 10–14,
2018, Proceedings, Part I 18. Springer International Publishing, 2019.

[2] Lu, Jiajun, Hussein Sibai, and Evan Fabry. Adversarial examples that fool detectors. arXiv preprint
arXiv:1712.02494(2017).

[3] Biggio, Battista, and Fabio Roli. Wild patterns: Ten years after the rise of adversarial machine learning.
Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security. 2018.
