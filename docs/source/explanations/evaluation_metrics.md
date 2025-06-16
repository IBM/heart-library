# Evaluation Metrics

Adversarial AI attacks should be evaluated in terms of their effects on the model performance as a function of attack
effort, such as size of adversarial perturbations or number of model queries. Key metrics for evaluating attack efficacy
include:

- _Image classification_ models are mainly evaluated using classification **accuracy**, or the percent of images
  correctly classified out of all evaluated images. Other classification metrics like False Positive Rate can be used to
  deepen insights into the evaluation.

- _Object detection_ models are mainly evaluated using **mean average precision (mAP)**, which is calculated as the
  difference between the ground truth object bounding boxes and the prediction boxes. Effect-specific metrics for
  hallucinations, disappearance, misclassification of objects, are also used to provide a more detailed evaluation.

Evaluation begins with the model’s baseline performance, also known as its **clean** (also called **benign**)
performance: for image classification models evaluation would start with **clean accuracy**, while for object detection
the evaluation would start with **clean mAP**.

The adversarial attack is then carried out, and the model’s performance is assessed again, this time under attack. This
second assessment uses the same metrics (accuracy or mAP), but now qualified as either **robust** or **adversarial**
performance:

- **Robust** performance is calculated after adversarial attack on all samples.

- **Adversarial** performance is calculated after adversarial attack on samples correctly predicted in the
  non-adversarial scenario.

Thus, for an image classification model, the difference between **clean accuracy** (before an evasion attack), and
**robust accuracy** (after the attack), would serve to indicate how robust the AI model is to withstand that type of
adversarial attack. The difference between **clean accuracy** and **adversarial accuracy** would similarly indicate the
model's adversarial robustness.
