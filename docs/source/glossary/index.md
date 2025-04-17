Glossary
=========
This Glossary servers to help HEART users with understanding Adversarial Robustness terms used throughout the documentation.

:::{glossary}
:sorted:

**Accuracy**
  - Accuracy: Items correct out of total items tested, used to evaluate classification models.

  - Accuracy, Clean: Standard accuracy, before adversarial attack.

  - Accuracy, Robust: Accuracy after adversarial attack on all samples.

  - Accuracy, Adversarial: Accuracy after adversarial attack on samples correctly predicted in non-adversarial scenario.

**Adversarial Perturbation Size**
  Usually defined by the norm of the perturbation matrix.

**mAP (mean Average Precision)**
  Used to evaluate object detection models (calculated difference between ground truth and prediction boxes).

**Adversarial Attacks**
  An adversarial attack in machine learning is a deliberate attempt to deceive or manipulate a model by injecting carefully crafted, often imperceptible, changes to its input data. The goal is to cause the model to make incorrect predictions or take unintended actions, often exploiting vulnerabilities in the model's learning process. [^1]

**Adversarial Robustness**
  Adversarial robustness refers to a model’s ability to resist being fooled or misled by an adversarial attack. [^2]

**Adversarial Robustness Toolkit (ART)**
  The Adversarial Robustness Toolbox (ART) is an open-source project, started by IBM, for machine learning security and has recently been donated to the Linux Foundation for AI (LFAI) by IBM as part of the Trustworthy AI tools. ART focuses on the threats of Evasion (change the model behavior with input modifications), Poisoning (control a model with training data modifications), Extraction (steal a model through queries) and Inference (attack the privacy of the training data). ART aims to support all popular ML frameworks, tasks, and data types and is under continuous development, lead by our team, to support both internal and external researchers and developers in defending AI against adversarial attacks and making AI systems more secure. [^3]

**Adversarial Training**
  Adversarial training uses a mixture of adversarial examples and benign samples to train an ML model aiming to increase its robustness against adversarial examples. [^4]

**AI Security**
  AI security involves protecting artificial intelligence systems from unauthorized access, misuse, and attacks, as well as ensuring the reliability and safety of AI models and their applications. It encompasses various aspects, including defending against cyber threats, addressing algorithmic bias, and mitigating potential harm from AI systems. [^5]

**Attack Failure**
  Attack failure refers to T&E experiments where the attack completes sucessfully but has not performed optimally, for example because of numerical underflow of the applied loss gradients or too few attack iterations. It is a key responsibility of T&E to validate that attacks have been applied correctly and have achieved their expected performance. [^6]

**Black-Box Attack**
  These attacks assume that the attacker operates with minimal, and sometimes no knowledge at all about the ML system. An adversary might have query access to the model, but they have no other information about how the model is trained.  These attacks are the most practical since they assume that the attacker has no knowledge of the AI system and utilizes system interfaces readily available for normal use.

**Certified Defense**
  A defense that has mathematical guarantees on model robustness after defending against an attack.

**CIFAR-10**
  CIFAR-10 is a widely used dataset in machine learning and computer vision for training and evaluating image classification algorithms. It consists of 60,000 small, 32x32 color images, divided into 10 classes. These classes represent common objects like airplanes, cars, birds, and more. The dataset is divided into a training set of 50,000 images and a test set of 10,000 images. [^7]

**Denial of Service (Object Detection)**
  Denial of service on object detection refers to input perturbations that lead the object detection model to predict an extremely large number of objects, which significantly slows down the processing of images or eventually results in recourse exhaustion. [^8]

**Evasion Attacks**
  Evasion attacks attempt to make a model output incorrect results by slightly perturbing the input data that is sent to the trained model. [^9]

**GRAPHITE**
  GRAPHITE is a patch attack that automatically constrains the size and shape of the patch so it can be applied with stickers, optimizes robustness of the attack to environmental physical variations such as viewpoint and lighting changes, and supports attacks in white- and black-box hard-label scenarios. [^10]

**Hardened Extension of the Adversarial Robustness Toolbox (HEART)**
  The Hardened Extension of the Adversarial Robustness Toolbox (HEART) is a modular open-source python library that provides AI developers and researchers with testing and evaluation (T&E) tools to assess AI model performance under adversarial attacks and improve model resiliency. [^11]

**HopSkipJump**
  HopSkipJump is a decision-based, black-box, evasion attack on a trained model that generates adversarial examples based solely on observing output labels returned by the targeted model. [^12]

**Interval Bound Propagation**
  Interval Bound Propagation (IBP) is a technique used in machine learning, particularly for verifying and training neural networks with provably robust guarantees. It involves using interval arithmetic to track the upper and lower bounds of a neural network's output when the input is uncertain or perturbed.

**Invisibility (Object Detection)**
  Invisibility or disappearance describes an attack goal for attacks on object detection or tracking models where a targeted object is not detected by the model anymore after applying the attack.

**Joint AI Test Infrastructure Capability (JATIC)**
  The Joint AI Test Infrastructure Capability (JATIC) program develops software products for AI Test & Evaluation (T&E) and AI Assurance. [^13]

**JPEG Compression**
  A JPEG compression defense is a technique used to mitigate the effects of adversarial attacks on image classification models. It involves compressing images using JPEG before feeding them to the model, effectively smoothing out or removing the fine-grained perturbations that adversarial attacks introduce. This helps the model to make more accurate predictions by focusing on the essential features of the image rather than being misled by the noise generated by the attack. [^14]

**Laser Beam Attack**
  The Laser Beam Attack uses greedy search to optimize localised, unbounded perturbations to images that reproduce the effect of laser pointers to image sensors like cameras to negatively influence the performance of ML models processing these images. [^15]

**MAITE**
  MAITE is a library of common types, protocols (a.k.a. structural subtypes), and utilities for the test and evaluation (T&E) of supervised machine learning models. It is being developed under the Joint AI T&E Infrastructure Capability (JATIC) program. [^16]

**Madry's Protocol**
  Madry's Protocol refers to the first successful adversarial training algorithm for image classification models. This protocol generated adversarial examples using Projected Gradient Descent and trains image classification models on a mixture of adversarial examples and benign samples. [^17]

**Mitigating Defense**
  A defense that mitigates the effects of adversarial attacks, but does not have any mathematical guarantees on the minimally provided model robustness. 

**Patch Attack**
  This attack generates adversarial patches that can be printed or projected and applied in the physical world to attack image and video processing ML models. [^18]

**Perturbation**
 - Perturbation Size: The difference between the original input and its adversarial version. For pictures usually calculated as a L2- or L_inf-norm of the differences of all pixels.
 - Perturbation Type: a small, intentional change or modification applied to input data, model parameters, or the model's training process. These changes are used to explore how sensitive a model is to variations in its inputs and to test the robustness of a model. Perturbations can be used for various purposes, including improving model robustness, detecting overfitting, and identifying vulnerabilities in the model.
 - Local Perturbation: Local perturbation refers to adversarial perturbations that only affect a subset of all input features. For example adversarial patches are local perturbations because they only cover a subset of all pixels of an input image.

**Physical Realizability**
  An adversarial attack is physically realizable if its adversarial perturbation can be materialized in the real world by, for example, printing the perturbation on paper, projecting it with a light or energy source, or transmit it as sound with a speaker.

**Post-Processing Defense**
  A post-processing defense in AI involves modifying the output of a machine learning model after the initial prediction is made to improve robustness and resilience against adversarial attacks. This can include techniques like thresholding, smoothing, and filtering to refine the model's output and reduce the impact of adversarial manipulations.

**Pre-Processing Defense**
  A pre-processing defense aims to modify the input (e.g. images) in attempt to remove, detect, or render ineffective the adversarial perturbation included in the input.

**Projected Gradient Descent (PGD)**
  A white-box evasion attack considered the gold-standard in robustness evaluation. [^19]

**Robust Accuracy**
  The accuracy of the model on adversarial examples created by an adversarial attack.

**Spatial Smoothing**
  Spatial smoothing is a technique that involves averaging or blurring values of nearby data points to reduce noise and high-frequency variations.[^20]

**Transfer Attacks**
  In a transfer attacks on AI the attacker crafts adversarial examples using an accessible model and uses these adversarial examples to attack another model. This allows the attacker to remain undetected during the crafting of the adversarial examples, use stronger attacks because of the detailed access to the accessible model, and attack the targeted model on the first access. [^21]
:::

## References [^Note]
These references were used to create the Glossary definitions above.

[^Note]: Many definitions in the HEART Glossary reference standard definitions set in [NIST AI 100-2e2025](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-2e2025.pdf) "Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations" created by [NIST Trustworthy and Responsible AI](https://airc.nist.gov/).

[^1]:  https://cltc.berkeley.edu/aml/#:~:text=What%20are%20adversarial%20attacks?,an%20%E2%80%9Cevasion%E2%80%9D%20attack).
[^2]:  https://research.ibm.com/blog/securing-ai-workflows-with-adversarial-robustness
[^3]:  https://research.ibm.com/projects/adversarial-robustness-toolbox
[^4]:  https://arxiv.org/abs/1706.06083
[^5]:  https://www.nsa.gov/AISC/#:~:text=AI%20Security%20is%20protecting%20AI%20systems%20from,integrity%2C%20and%20availability%20of%20information%20and%20services.
[^6]:  https://arxiv.org/abs/2106.09947
[^7]:  https://archive.ics.uci.edu/dataset/691/cifar+10
[^8]:  https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Overload_Latency_Attacks_on_Object_Detection_for_Edge_Devices_CVPR_2024_paper.pdf
[^9]:  https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.ibm.com/docs/en/watsonx/saas%3Ftopic%3Datlas-evasion-attack&ved=2ahUKEwiTgvPdy9iMAxWrMlkFHcBuNQoQFnoECBYQAw&usg=AOvVaw1G-6ZTdI9c5XgXVahKL0Ob
[^10]:  https://arxiv.org/abs/2002.07088
[^11]:  HEART-library Documentation
[^12]:  https://arxiv.org/abs/1904.02144
[^13]:  https://cdao.pages.jatic.net/public/
[^14]:  https://istc-arsa.iisp.gatech.edu/defending-ai-jpeg.html
[^15]:  https://arxiv.org/abs/2103.06504
[^16]:  https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/mit-ll-ai-technology/maite&ved=2ahUKEwjezLjTxNiMAxV4FFkFHS8SK9cQFnoECBIQAw&usg=AOvVaw2a8ZdtpQcU0PIi6jB86mNp
[^17]:  https://arxiv.org/abs/1706.06083
[^18]:  https://art360.res.ibm.com/resources#attacks
[^19]:  https://arxiv.org/abs/1706.06083
[^20]:  https://arxiv.org/abs/2105.12639
[^21]: https://www.paloaltonetworks.com/cyberpedia/what-are-adversarial-attacks-on-AI-Machine-Learning#:~:text=Transfer%20attacks%20represent%20a%20unique%20challenge%20in,their%20adaptation%20to%20attack%20other%20AI%20models.