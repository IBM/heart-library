'''
HEART Gradio Example App

To run: 
- clone the repository
- execute: gradio examples/gradio_app.py or python examples/gradio_app.py
- navigate to local URL e.g. http://127.0.0.1:7860
'''

import torch
import numpy as np
import pandas as pd
# from carbon_theme import Carbon

import gradio as gr
import os
from typing import Tuple, Dict, Any

css = """
.input-image { margin: auto !important }
.small-font span{
 font-size: 0.6em;
}
.df-padding {
    padding-left: 50px !important;
    padding-right: 50px !important;
}

.output-image, img {
    border-radius: 0px !important;
    margin: auto !important;
} 
"""

REQUESTS_TIMEOUT = 300
COCO_LABELS = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def update_patch_sliders(*args):
    from typing import Dict
    
    x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image = args
    
    if dataset_type == "Example XView":
        from datasets import load_dataset
        from heart_library.utils import process_inputs_for_art
        from torchvision import transforms
        
        jatic_dataset = load_dataset("CDAO/xview-subset-classification", split="test[0:2]")
        
        IMAGE_H, IMAGE_W = 224, 224

        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor()
        ])

        jatic_dataset = jatic_dataset.map(lambda x: {"image": preprocess(x["image"]), "label": x["label"]})
        images, _, _ = process_inputs_for_art(jatic_dataset)
        
        # transform pixel value 0-255 for visual purpose
        images = (images[0]*255).transpose(1,2,0).astype(np.uint8)
        
    elif dataset_type=="huggingface":
        pass
    elif dataset_type=="torchvision":
        pass  
    elif dataset_type=="Example CIFAR10":
        from datasets import load_dataset
        from heart_library.utils import process_inputs_for_art
        jatic_dataset = load_dataset("cifar10", split="test[0:10]") 
        images, _, _ = process_inputs_for_art(jatic_dataset)
    elif dataset_type=="COCO":
        from datasets import Dataset
        from datasets import load_dataset
        from torchvision.transforms import transforms
        from functools import partial
        import requests
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput
        from typing import Dict, Tuple, Any
        from secrets import SystemRandom
        from heart_library.utils import process_inputs_for_art

        NUM_SAMPLES = 1

        data = load_dataset("detection-datasets/coco", split="val", streaming=True)
        sample_data = data.take(NUM_SAMPLES)
        
        preprocess = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor()
        ])

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        def transform_bbox(object, width, height):
            '''
            Scale the bboxes in line with scaled images
            '''
            x_scale = 800 / width
            y_scale = 800 / height
            transformed_bbox = []
            for bbox in object["bbox"]:
                xmin, ymin, xmax, ymax = bbox
                x1 = int(np.round(xmin * x_scale))
                y1 = int(np.round(ymin * y_scale))
                x2 = int(np.round(xmax * x_scale))
                y2 = int(np.round(ymax * y_scale))
                transformed_bbox.append([x1, y1, x2, y2])
            object["bbox"] = transformed_bbox
            return object


        data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        data = data.map(lambda x: {"image": preprocess(x["image"]), "objects": transform_bbox(x["objects"], x["width"], x["height"])})
        
        class CocoDetectionDataset:
            def __init__(self, images, boxes, scores, labels):
                # COCO dataset labels
                response = requests.get("https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt",
                                        timeout=REQUESTS_TIMEOUT)
                dataset_labels = response.text.split("\n")
                self._images = images
                self._boxes = boxes
                self._scores = scores
                self._labels = [[COCO_LABELS.index(dataset_labels[label]) for label in item] for item in labels]
                
            def __len__(self)->int:
                return len(self._images)
            
            def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                image = self._images[ind]
                detection = {"boxes": self._boxes[ind],
                            "scores": self._scores[ind],
                            "labels": self._labels[ind]}
                detection = JaticPyTorchObjectDetectionOutput(detection)
                return (image, detection, {})

        labels = [item['objects']['category'] for item in data]
        boxes = [np.array(item['objects']['bbox']) for item in data]
        scores = [np.array([SystemRandom().uniform(0.8, 1) for _ in range(len(item['objects']['category']))]) for item in data]
        images = [item['image'] for item in data]   

        jatic_dataset = CocoDetectionDataset(images, boxes, scores, labels)
        images, _, _ = process_inputs_for_art(jatic_dataset)
        images = (images[0]*255).transpose(1,2,0).astype(np.uint8)
        
    elif dataset_type=="VisDrone":
        from datasets import Dataset
        from datasets import load_dataset
        from torchvision.transforms import transforms
        from functools import partial
        import requests
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput
        from secrets import SystemRandom
        from heart_library.utils import process_inputs_for_art
        
        NUM_SAMPLES = 5

        data = load_dataset("Voxel51/VisDrone2019-DET", split="train", streaming=True)
        sample_data = data.take(NUM_SAMPLES)

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        sample_data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        
        IMAGE_H, IMAGE_W = 800, 800
        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor()
        ])
        sample_data = sample_data.map(lambda x: {"image": preprocess(x["image"]), "label": None})
        images, _, _ = process_inputs_for_art(sample_data)
        images = (images[0]*255).transpose(1,2,0).astype(np.uint8)
        
    if isinstance(images, Dict) and "image" in images:
        images = images['image']
    
    if isinstance(images, list):
        images = images[0]
         
    height = images.shape[0]
    width = images.shape[1]
    
    max_patch = min(height, width)
    if patch_dim > max_patch:
        patch_dim = max_patch
    
    max_x = width - (patch_dim) 
    max_y = height - (patch_dim)
    
    max_x = max_x if max_x >= 0 else 0
    max_y = max_y if max_y >= 0 else 0
    
    if x_location > max_x:
        x_location = max_x
    if y_location > max_y:
        y_location = max_y
    
    return [gr.Slider(maximum=max_patch, step=1), gr.Slider(maximum=max_x, value=x_location, step=1), gr.Slider(maximum=max_y, value=y_location, step=1)]

def preview_patch_location(*args):
    '''
    Create a gallery of images with a sample patch applied
    '''
    import cv2

    x_location, y_location, patch_dim = int(args[0]), int(args[1]), int(args[2])

    dataset_type = args[-4]
    dataset_path = args[-3]
    dataset_split = args[-2]
    image = args[-1]

    if dataset_type == "Example XView":
        from datasets import load_dataset
        from heart_library.utils import process_inputs_for_art
        from torchvision import transforms
        
        jatic_dataset = load_dataset("CDAO/xview-subset-classification", split="test[0:2]")
        
        IMAGE_H, IMAGE_W = 224, 224

        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor()
        ])

        jatic_dataset = jatic_dataset.map(lambda x: {"image": preprocess(x["image"]), "label": x["label"]})
        images, _, _ = process_inputs_for_art(jatic_dataset) 
        
        # transform pixel value 0-255 for visual purpose
        images = (images[0]*255).transpose(1,2,0).astype(np.uint8)
        
    elif dataset_type=="huggingface":
        pass
    elif dataset_type=="torchvision":
        pass
    elif dataset_type=="Example CIFAR10":
        from datasets import load_dataset
        from heart_library.utils import process_inputs_for_art
        jatic_dataset = load_dataset("cifar10", split="test[0:10]") 
        images, _, _ = process_inputs_for_art(jatic_dataset)
    elif dataset_type=="COCO":
        from datasets import Dataset
        from datasets import load_dataset
        from torchvision.transforms import transforms
        from functools import partial
        import requests
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput
        from typing import Dict, Tuple, Any
        from secrets import SystemRandom
        from heart_library.utils import process_inputs_for_art

        NUM_SAMPLES = 1

        data = load_dataset("detection-datasets/coco", split="val", streaming=True)
        sample_data = data.take(NUM_SAMPLES)
        
        preprocess = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor()
        ])

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        def transform_bbox(object, width, height):
            '''
            Scale the bboxes in line with scaled images
            '''
            x_scale = 800 / width
            y_scale = 800 / height
            transformed_bbox = []
            for bbox in object["bbox"]:
                xmin, ymin, xmax, ymax = bbox
                x1 = int(np.round(xmin * x_scale))
                y1 = int(np.round(ymin * y_scale))
                x2 = int(np.round(xmax * x_scale))
                y2 = int(np.round(ymax * y_scale))
                transformed_bbox.append([x1, y1, x2, y2])
            object["bbox"] = transformed_bbox
            return object


        data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        data = data.map(lambda x: {"image": preprocess(x["image"]), "objects": transform_bbox(x["objects"], x["width"], x["height"])})
        
        class CocoDetectionDataset:
            def __init__(self, images, boxes, scores, labels):
                # COCO dataset labels
                response = requests.get("https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt",
                                        timeout=REQUESTS_TIMEOUT)
                dataset_labels = response.text.split("\n")
                self._images = images
                self._boxes = boxes
                self._scores = scores
                self._labels = [[COCO_LABELS.index(dataset_labels[label]) for label in item] for item in labels]
                
            def __len__(self)->int:
                return len(self._images)
            
            def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                image = self._images[ind]
                detection = {"boxes": self._boxes[ind],
                            "scores": self._scores[ind],
                            "labels": self._labels[ind]}
                detection = JaticPyTorchObjectDetectionOutput(detection)
                return (image, detection, {})

        labels = [item['objects']['category'] for item in data]
        boxes = [np.array(item['objects']['bbox']) for item in data]
        scores = [np.array([SystemRandom().uniform(0.8, 1) for _ in range(len(item['objects']['category']))]) for item in data]
        images = [item['image'] for item in data]   

        jatic_dataset = CocoDetectionDataset(images, boxes, scores, labels)
        images, _, _ = process_inputs_for_art(jatic_dataset)
        images = (images[0]*255).transpose(1,2,0).astype(np.uint8)
        
    elif dataset_type=="VisDrone":
        from datasets import Dataset
        from datasets import load_dataset
        from torchvision.transforms import transforms
        from functools import partial
        import requests
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput
        from typing import Dict, Tuple, Any
        from secrets import SystemRandom
        from heart_library.utils import process_inputs_for_art
        
        NUM_SAMPLES = 5

        data = load_dataset("Voxel51/VisDrone2019-DET", split="train", streaming=True)
        sample_data = data.take(NUM_SAMPLES)

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        sample_data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        
        IMAGE_H, IMAGE_W = 800, 800
        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor()
        ])
        sample_data = sample_data.map(lambda x: {"image": preprocess(x["image"]), "label": None})
        images, _, _ = process_inputs_for_art(sample_data)
        images = (images[0]*255).transpose(1,2,0).astype(np.uint8)
        
    if isinstance(images, Dict) and "image" in images:
        image = image['image']
    
    if isinstance(images, list):
        images = images[0]

    p0 = x_location, y_location
    p1 =  x_location + (patch_dim-1), y_location + (patch_dim-1)
    
    images = cv2.rectangle(cv2.UMat(images), p0, p1, (255,165,0), cv2.FILLED).get()
    
    return images

def extract_predictions(predictions_, conf_thresh):
    # Get the predicted class
    predictions_class = [COCO_LABELS[i] for i in list(predictions_.labels)]
    #  print("\npredicted classes:", predictions_class)
    if len(predictions_class) < 1:
        return [], [], []
    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_.boxes)]

    # Get the predicted prediction score
    predictions_score = list(predictions_.scores)
    # print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = conf_thresh
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
    if len(predictions_t) > 0:
        predictions_t = predictions_t  # [-1] #indices where score over threshold
    else:
        # no predictions esxceeding threshold
        return [], [], []
    # predictions in score order
    predictions_boxes = [predictions_boxes[i] for i in predictions_t]
    predictions_class = [predictions_class[i] for i in predictions_t]
    predictions_scores = [predictions_score[i] for i in predictions_t]
    return predictions_class, predictions_boxes, predictions_scores

def plot_image_with_boxes(img, boxes, pred_cls, title):
    import cv2  
    text_size = 1
    text_th = 2
    rect_th = 1
    img = img * 255

    sections = []
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      color=(0, 255, 0), thickness=rect_th)
        # Write the prediction class
        cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)
        sections.append( ((int(boxes[i][0][0]),
                           int(boxes[i][0][1]),
                           int(boxes[i][1][0]), 
                           int(boxes[i][1][1])), (pred_cls[i])) )
    

    return img.astype(np.uint8)
    
def filter_boxes(predictions, conf_thresh):
    dictionary = {}

    boxes_list = []
    scores_list = []
    labels_list = []

    for i in range(len(predictions[0]["boxes"])):
        score = predictions[0]["scores"][i]
        if score >= conf_thresh:
            boxes_list.append(predictions[0]["boxes"][i])
            scores_list.append(predictions[0]["scores"][[i]])
            labels_list.append(predictions[0]["labels"][[i]])
            
    dictionary["boxes"] = np.vstack(boxes_list)
    dictionary["scores"] = np.hstack(scores_list)
    dictionary["labels"] = np.hstack(labels_list)

    y = [dictionary]

    return y

def basic_cifar10_model():
    '''
    Load an example CIFAR10 model
    '''
    from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
    
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    path = './'
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

            # pylint: disable=W0221
            # disable pylint because of API requirements for function
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
                return x

    # Define the network
    model = Model()

    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Get classifier
    jptc = JaticPyTorchClassifier(
        model=model, loss=loss_fn, optimizer=optimizer, input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 255),
        preprocessing=(0.0, 255)
    )
    return jptc

def det_evasion_evaluate(*args):
    '''
    Run a detection task evaluation
    '''
    
    attack = args[0]
    model_type = args[1]
    
    box_thresh = args[-3]
    dataset_type = args[-2]
    image = args[-1]
    
    if dataset_type == "COCO":
        from datasets import Dataset
        from datasets import load_dataset
        from torchvision.transforms import transforms
        from functools import partial
        import requests
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput
        from typing import Dict, Tuple, Any
        from secrets import SystemRandom
        from heart_library.utils import process_inputs_for_art

        NUM_SAMPLES = 5

        data = load_dataset("detection-datasets/coco", split="val", streaming=True)
        sample_data = data.take(NUM_SAMPLES)
        
        preprocess = transforms.Compose([
            transforms.Resize((800, 800)),
            # transforms.CenterCrop(800),
            transforms.ToTensor()
        ])

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        def transform_bbox(object, width, height):
            '''
            Scale the bboxes in line with scaled images
            '''
            x_scale = 800 / width
            y_scale = 800 / height
            transformed_bbox = []
            for bbox in object["bbox"]:
                xmin, ymin, xmax, ymax = bbox
                x1 = int(np.round(xmin * x_scale))
                y1 = int(np.round(ymin * y_scale))
                x2 = int(np.round(xmax * x_scale))
                y2 = int(np.round(ymax * y_scale))
                transformed_bbox.append([x1, y1, x2, y2])
            object["bbox"] = transformed_bbox
            return object


        data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        data = data.map(lambda x: {"image": preprocess(x["image"]), "objects": transform_bbox(x["objects"], x["width"], x["height"])})
        
        class CocoDetectionDataset:
            def __init__(self, images, boxes, scores, labels):
                # COCO dataset labels
                response = requests.get("https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt",
                                        timeout=REQUESTS_TIMEOUT)
                dataset_labels = response.text.split("\n")
                self._images = images
                self._boxes = boxes
                self._scores = scores
                self._labels = [[COCO_LABELS.index(dataset_labels[label]) for label in item] for item in labels]
                
            def __len__(self)->int:
                return len(self._images)
            
            def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                image = self._images[ind]
                detection = {"boxes": self._boxes[ind],
                            "scores": self._scores[ind],
                            "labels": self._labels[ind]}
                detection = JaticPyTorchObjectDetectionOutput(detection)
                return (image, detection, {})

        labels = [item['objects']['category'] for item in data]
        boxes = [np.array(item['objects']['bbox']) for item in data]
        scores = [np.array([SystemRandom().uniform(0.8, 1) for _ in range(len(item['objects']['category']))]) for item in data]
        images = [item['image'] for item in data]   

        jatic_dataset = CocoDetectionDataset(images, boxes, scores, labels)
        images, _, _ = process_inputs_for_art(jatic_dataset)
        groundtruth_detections = [item[1] for item in jatic_dataset]
        
    elif dataset_type == "VisDrone":
        from datasets import Dataset
        from datasets import load_dataset
        from torchvision.transforms import transforms
        from functools import partial
        import requests
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetectionOutput
        from typing import Dict, Tuple, Any
        from secrets import SystemRandom
        from heart_library.utils import process_inputs_for_art
        
        NUM_SAMPLES = 5

        data = load_dataset("Voxel51/VisDrone2019-DET", split="train", streaming=True)
        sample_data = data.take(NUM_SAMPLES)

        def gen_from_iterable_dataset(iterable_ds):
            yield from iterable_ds

        sample_data = Dataset.from_generator(partial(gen_from_iterable_dataset, sample_data), features=sample_data.features)
        
        IMAGE_H, IMAGE_W = 800, 800
        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor()
        ])
        sample_data = sample_data.map(lambda x: {"image": preprocess(x["image"]), "label": None})
        images, _, _ = process_inputs_for_art(sample_data)
        groundtruth_detections = []
        
    if model_type == "DeTR":
        from heart_library.estimators.object_detection import JaticPyTorchObjectDetector
        
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        NUMBER_CHANNELS = 3
        INPUT_SHAPE = (NUMBER_CHANNELS, 800, 800)

        '''
        Define the detector
        '''
        detector = JaticPyTorchObjectDetector(model_type="detr_resnet50",
                                              device_type='cpu',
                                            input_shape=INPUT_SHAPE,
                                            clip_values=(0, 1), 
                                            attack_losses=( "loss_ce",
                                                "loss_bbox",
                                                "loss_giou",),
                                            channels_first=True, 
                                            preprocessing=(MEAN, STD))
    
    if attack=="PGD":
        
        from art.attacks.evasion import ProjectedGradientDescent
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics import HeartMAPMetric
        from copy import deepcopy
        
        pgd_attack = ProjectedGradientDescent(estimator=detector, max_iter=args[7], eps=args[8],
                                                 eps_step=args[9], targeted=args[10]!="")
        attack = JaticAttack(pgd_attack, norm=2)
        
        benign_detections = detector(images)
        
        # for Visdrone dataset which does not contain targets
        if len(groundtruth_detections) == 0:
            groundtruth_detections = benign_detections
            
            class ImageDataset:
                def __init__(self, images, groundtruth, threshold=0.8):
                    self.images = images
                    self.groundtruth = groundtruth
                    self.threshold = threshold
                    
                def __len__(self)->int:
                    return len(self.images)
                
                def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                    image = self.images[ind]
                    filtered_detection = self.groundtruth[ind]
                    filtered_detection.boxes = filtered_detection.boxes[filtered_detection.scores>self.threshold]
                    filtered_detection.labels = filtered_detection.labels[filtered_detection.scores>self.threshold]
                    filtered_detection.scores = filtered_detection.scores[filtered_detection.scores>self.threshold]
                    return image, filtered_detection, {}
                
            jatic_dataset = ImageDataset(images, deepcopy(benign_detections), threshold=0.9)
            
        map_metric = HeartMAPMetric()
        map_metric.update(benign_detections, groundtruth_detections)
        benign_map = map_metric.compute()
        benign_map_50 = benign_map["map_50"].item()
        
        if args[-4]!="":
            class TargetedImageDataset:
                def __init__(self, images, groundtruth, target_label):
                    self.images = images
                    self.groundtruth = groundtruth
                    self.target_label = target_label
                    
                def __len__(self)->int:
                    return len(self.images)
                
                def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                    image = self.images[ind]
                    targeted_detection = self.groundtruth[ind]
                    targeted_detection.labels = [self.target_label]*len(targeted_detection.labels)
                    return image, targeted_detection, {}
                
            jatic_dataset = TargetedImageDataset(images, deepcopy(groundtruth_detections), args[-4])
        
        
        adv_x, _, metadata = attack(jatic_dataset)
        adv_detections = detector(adv_x)
        map_metric.update(adv_detections, groundtruth_detections)
        adv_map = map_metric.compute()
        adv_map_50 = adv_map["map_50"].item()
        
        mean_delta = np.array([item['delta'] for item in metadata]).mean()
        
        out_imgs = []
        for i in range(len(adv_detections)):
            preds_orig = extract_predictions(adv_detections[i], box_thresh)
            out_img = plot_image_with_boxes(img=adv_x[0][i].transpose(1,2,0).copy(),
                                        boxes=preds_orig[1], pred_cls=preds_orig[0], title="Detections")
            if out_img.max() > 1:
                out_img = out_img.astype(np.uint8)
            else:
                out_img = out_img.astype(np.float32)
            out_imgs.append(out_img)
    
        out_imgs_benign = []
        for i in range(len(benign_detections)):
            preds_benign = extract_predictions(benign_detections[i], box_thresh)
            out_img = plot_image_with_boxes(img=images[i].transpose(1,2,0).copy(),
                                        boxes=preds_benign[1], pred_cls=preds_benign[0], title="Detections")
            if out_img.max() > 1:
                out_img = out_img.astype(np.uint8)
            else:
                out_img = out_img.astype(np.float32)
            out_imgs_benign.append(out_img)
        
        return [out_imgs_benign, out_imgs, benign_map_50, adv_map_50, mean_delta]

    elif attack=="Adversarial Patch":
        from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics import HeartMAPMetric
        from copy import deepcopy

        batch_size = 16
        scale_min = 0.3
        scale_max = 1.0
        rotation_max = 0
        learning_rate = 5000.

        patch_attack = AdversarialPatchPyTorch(estimator=detector, rotation_max=rotation_max, patch_location=(args[8], args[9]),
                            scale_min=scale_min, scale_max=scale_max, patch_type='square',
                            learning_rate=learning_rate, max_iter=args[7], batch_size=batch_size,
                            patch_shape=(3, args[10], args[10]), verbose=True, targeted=args[-4]!="")
        
        attack = JaticAttack(patch_attack, norm=2)
        
        benign_detections = detector(images)
        
        # for Visdrone dataset which does not contain targets
        if len(groundtruth_detections) == 0:
            groundtruth_detections = benign_detections
            
            class ImageDataset:
                def __init__(self, images, groundtruth, threshold=0.8):
                    self.images = images
                    self.groundtruth = groundtruth
                    self.threshold = threshold
                    
                def __len__(self)->int:
                    return len(self.images)
                
                def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                    image = self.images[ind]
                    detection = self.groundtruth[ind]
                    return image, detection, {}
                
            jatic_dataset = ImageDataset(images, deepcopy(benign_detections), threshold=0.9)
        else:
            # in the case of COCO
            # As Adversarial patch requires targets to all be same length, use the predicted detections
            jatic_dataset._boxes = [detection.boxes for detection in benign_detections]
            jatic_dataset._scores = [detection.scores for detection in benign_detections]
            jatic_dataset._labels = [detection.labels for detection in benign_detections]
            
        map_metric = HeartMAPMetric()
        map_metric.update(benign_detections, groundtruth_detections)
        benign_map = map_metric.compute()
        benign_map_50 = benign_map["map_50"].item()
        
        if args[-4]!="":
            class TargetedImageDataset:
                def __init__(self, images, groundtruth, target_label):
                    self.images = images
                    self.groundtruth = groundtruth
                    self.target_label = target_label
                    
                def __len__(self)->int:
                    return len(self.images)
                
                def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                    image = self.images[ind]
                    targeted_detection = self.groundtruth[ind]
                    targeted_detection.labels = [self.target_label]*len(targeted_detection.labels)
                    return image, targeted_detection, {}
                
            jatic_dataset = TargetedImageDataset(images, deepcopy(benign_detections), args[-4])
        
        adv_x, _, metadata = attack(jatic_dataset)
        adv_detections = detector(adv_x)
        map_metric.update(adv_detections, groundtruth_detections)
        adv_map = map_metric.compute()
        adv_map_50 = adv_map["map_50"].item()
        
        out_imgs_adv = []
        for i in range(len(adv_detections)):
            preds_orig = extract_predictions(adv_detections[i], box_thresh)
            out_img = plot_image_with_boxes(img=adv_x[0][i].transpose(1,2,0).copy(),
                                        boxes=preds_orig[1], pred_cls=preds_orig[0], title="Detections")
            if out_img.max() > 1:
                out_img = out_img.astype(np.uint8)
            else:
                out_img = out_img.astype(np.float32)
            out_imgs_adv.append(out_img)
    
        out_imgs_benign = []
        for i in range(len(benign_detections)):
            preds_benign = extract_predictions(benign_detections[i], box_thresh)
            out_img = plot_image_with_boxes(img=images[i].transpose(1,2,0).copy(),
                                        boxes=preds_benign[1], pred_cls=preds_benign[0], title="Detections")
            if out_img.max() > 1:
                out_img = out_img.astype(np.uint8)
            else:
                out_img = out_img.astype(np.float32)
            out_imgs_benign.append(out_img)
            
        patch = metadata[0]["patch"]
        patch_mask = metadata[0]["mask"]
        
        patch_image = ((patch) * patch_mask).transpose(1,2,0)
        if patch_image.max() > 1:
            patch_image = patch_image.astype(np.uint8)
        else:
            patch_image = patch_image.astype(np.float32)
            
        return [out_imgs_benign, out_imgs_adv, benign_map_50, adv_map_50, patch_image]

def clf_evasion_evaluate(*args):
    '''
    Run a classification task evaluation
    '''
    
    attack = args[0]
    model_type = args[1]
    model_path = args[2]
    model_channels = args[3]
    model_height = args[4]
    model_width = args[5]
    model_clip = args[6]
    
    dataset_type = args[-4]
    dataset_path = args[-3]
    dataset_split = args[-2]
    image = args[-1]
    
    if dataset_type == "Example XView":
        from datasets import load_dataset
        from heart_library.utils import process_inputs_for_art
        from torchvision import transforms
        
        clf_labels = ['Building', 'Construction Site', 'Engineering Vehicle', 'Fishing Vessel', 'Oil Tanker', 'Vehicle Lot']
        jatic_dataset = load_dataset("CDAO/xview-subset-classification", split="test[0:100]")
        
        IMAGE_H, IMAGE_W = 224, 224

        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor()
        ])

        jatic_dataset = jatic_dataset.map(lambda x: {"image": preprocess(x["image"]), "label": x["label"]})
        images, groundtruth, _ = process_inputs_for_art(jatic_dataset) 
    elif dataset_type=="huggingface":
        pass
    elif dataset_type=="torchvision":
        pass
         
    elif dataset_type=="Example CIFAR10":
        from datasets import load_dataset
        from heart_library.utils import process_inputs_for_art
        
        clf_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        jatic_dataset = load_dataset("cifar10", split="test[0:10]") 
        images, groundtruth, _ = process_inputs_for_art(jatic_dataset)
        
    if model_type == "Example CIFAR10":
        jptc = basic_cifar10_model()  
    elif model_type == "Example XView":
        import torchvision
        from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
        classes = {
            0:'Building',
            1:'Construction Site',
            2:'Engineering Vehicle',
            3:'Fishing Vessel',
            4:'Oil Tanker',
            5:'Vehicle Lot'
        }
        model = torchvision.models.resnet18(False)
        num_ftrs = model.fc.in_features 
        model.fc = torch.nn.Linear(num_ftrs, len(classes.keys())) 
        model.load_state_dict(torch.load('./utils/resources/models/xview_model.pt'))
        _ = model.eval()
        jptc = JaticPyTorchClassifier(
            model=model, loss = torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224),
            nb_classes=len(classes), clip_values=(0, 1)
        )
    elif model_type == "torchvision":
        from torchvision.models import resnet18, ResNet18_Weights
        from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
        
        resnet_model = resnet18(ResNet18_Weights)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        jptc = JaticPyTorchClassifier(
            model=resnet_model, loss=loss_fn, input_shape=(model_channels, model_height, model_width), 
            nb_classes=1000, clip_values=(0, model_clip)
        )
    elif model_type == "huggingface":
        from heart_library.estimators.classification.pytorch import JaticPyTorchClassifier
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        jptc = JaticPyTorchClassifier(
            model=model_path, loss=loss_fn, input_shape=(model_channels, model_height, model_width),
            nb_classes=1000, clip_values=(0, model_clip), preprocessing=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            provider="huggingface"
        )
    
    if attack=="PGD":
        from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics import AccuracyPerturbationMetric
        from torch.nn.functional import softmax
        
        pgd_attack = ProjectedGradientDescentPyTorch(estimator=jptc, max_iter=args[7], eps=args[8],
                                                 eps_step=args[9], targeted=args[10]!="")
        attack = JaticAttack(pgd_attack, norm=2)
        
        preds = jptc(images)
        
        # override jatic_dataset with targeted dataset args[10]
        if args[11]!="":
            class TargetedImageDataset:
                def __init__(self, images, target):
                    self._images = images
                    self._target = target
                def __len__(self)->int:
                    return len(self._images)
                def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                    image = self._images[ind]
                    return image, self._target, {}
            
            images = TargetedImageDataset(images, args[10])
        
        x_adv, _, metadata = attack(images)
        adv_preds = jptc(x_adv)
        
        metric = AccuracyPerturbationMetric(preds, metadata)
        metric.update(adv_preds, groundtruth)
        result = metric.compute()

        # convert numpy.float64 to python float for gradio>=4.24.0 compatibility
        clean_accuracy = float(result["clean_accuracy"])
        robust_accuracy = float(result["robust_accuracy"])
        perturbation_added = float(result["mean_delta"])
        
        
        # transform to probabilities
        preds = softmax(torch.from_numpy(preds[0]), dim=1)
        labels = {}
        for i, label in enumerate(clf_labels):
            labels[label] = preds[0][i]
            
        adv_preds = softmax(torch.from_numpy(adv_preds[0]), dim=1)
        adv_labels = {}
        for i, label in enumerate(clf_labels):
            adv_labels[label] = adv_preds[0][i]

        adv_imgs = [img.transpose(1,2,0) for img in x_adv[0]]
        
        image_list = []

        for i, img in enumerate(images):
            if isinstance(img, Tuple):
                if img[0].max()>1:
                    image_list.append((img[0].transpose(1,2,0).astype(np.uint8), str(clf_labels[np.argmax(preds[i]).item()]) ))
                else:
                    image_list.append((img[0].transpose(1,2,0).astype(np.float32), str(clf_labels[np.argmax(preds[i]).item()]) ))
            else:
                if img.max()>1:
                    image_list.append((img.transpose(1,2,0).astype(np.uint8), str(clf_labels[np.argmax(preds[i]).item()]) ))
                else:
                    image_list.append((img.transpose(1,2,0).astype(np.float32), str(clf_labels[np.argmax(preds[i]).item()]) ))
        
        adv_image_list = []
        for i, img in enumerate(adv_imgs):
            if img.max()>1:
                adv_image_list.append((img.astype(np.uint8), str(clf_labels[np.argmax(adv_preds[i]).item()]) ))
            else:
                adv_image_list.append((img.astype(np.float32), str(clf_labels[np.argmax(adv_preds[i]).item()]) ))
            
        
        
        return [image_list, labels, adv_image_list, adv_labels, clean_accuracy, robust_accuracy, perturbation_added]

    elif attack=="Adversarial Patch":
        from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
        from heart_library.attacks.attack import JaticAttack
        from heart_library.metrics import AccuracyPerturbationMetric
        from torch.nn.functional import softmax
        
        batch_size = 16
        scale_min = 0.3
        scale_max = 1.0
        rotation_max = 0
        learning_rate = 5000.

        patch_attack = AdversarialPatchPyTorch(estimator=jptc, rotation_max=rotation_max, patch_location=(args[8], args[9]),
                            scale_min=scale_min, scale_max=scale_max, patch_type='square',
                            learning_rate=learning_rate, max_iter=args[7], batch_size=batch_size,
                            patch_shape=(3, args[10], args[10]), verbose=False, targeted=args[11]!="")
        
        attack = JaticAttack(patch_attack, norm=2)
        
        preds = jptc(images)
        
        # override jatic_dataset with targeted dataset args[10]
        if args[11]!="":
            class TargetedImageDataset:
                def __init__(self, images, target):
                    self._images = images
                    self._target = target
                def __len__(self)->int:
                    return len(self._images)
                def __getitem__(self, ind: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
                    image = self._images[ind]
                    return image, self._target, {}
            
            images = TargetedImageDataset(images, args[11])
        
        x_adv, _, metadata = attack(images)
        adv_preds = jptc(x_adv)
        
        metric = AccuracyPerturbationMetric(preds, metadata)
        metric.update(adv_preds, groundtruth)
        result = metric.compute()

        # convert numpy.float64 to python float for gradio>=4.24.0 compatibility
        clean_accuracy = float(result["clean_accuracy"])
        robust_accuracy = float(result["robust_accuracy"])
        perturbation_added = float(result["mean_delta"])
        
        # transform to probabilities
        preds = softmax(torch.from_numpy(preds[0]), dim=1)
        
        labels = {}
        for i, label in enumerate(clf_labels):
            labels[label] = preds[0][i]
            
        adv_preds = softmax(torch.from_numpy(adv_preds[0]), dim=1)
        adv_labels = {}
        for i, label in enumerate(clf_labels):
            adv_labels[label] = adv_preds[0][i]

        # convert numpy.float64 to python float for gradio>=4.24.0 compatibility
        clean_accuracy = float(clean_accuracy)
        robust_accuracy = float(robust_accuracy)
        perturbation_added = float(perturbation_added)
        
        adv_imgs = [img.transpose(1,2,0) for img in x_adv[0]]
        
        image_list = []

        for i, img in enumerate(images):
            if isinstance(img, Tuple):
                if img[0].max()>1:
                    image_list.append((img[0].transpose(1,2,0).astype(np.uint8), str(clf_labels[np.argmax(preds[i]).item()]) ))
                else:
                    image_list.append((img[0].transpose(1,2,0).astype(np.float32), str(clf_labels[np.argmax(preds[i]).item()]) ))
            else:
                if img.max()>1:
                    image_list.append((img.transpose(1,2,0).astype(np.uint8), str(clf_labels[np.argmax(preds[i]).item()]) ))
                else:
                    image_list.append((img.transpose(1,2,0).astype(np.float32), str(clf_labels[np.argmax(preds[i]).item()]) ))
        
        adv_image_list = []
        for i, img in enumerate(adv_imgs):
            if img.max()>1:
                adv_image_list.append((img.astype(np.uint8), str(clf_labels[np.argmax(adv_preds[i]).item()]) ))
            else:
                adv_image_list.append((img.astype(np.float32), str(clf_labels[np.argmax(adv_preds[i]).item()]) ))
            
        patch = metadata[0]["patch"]
        patch_mask = metadata[0]["mask"]
        
        if patch.max()>1:
            patch_image = ((patch) * patch_mask).transpose(1,2,0).astype(np.uint8)
        else:
            patch_image = ((patch) * patch_mask).transpose(1,2,0).astype(np.float32)
                
        
        return [image_list, labels, adv_image_list, adv_labels, clean_accuracy, robust_accuracy, patch_image]
            
def show_model_params(model_type):
    '''
    Show model parameters based on selected model type
    '''
    if model_type!="Example CIFAR10" and model_type!="Example XView":
        return gr.Column(visible=True)
    return gr.Column(visible=False)
    
def show_dataset_params(dataset_type):
    '''
    Show dataset parameters based on dataset type
    '''
    if dataset_type=="Example CIFAR10" or dataset_type=="Example XView":
        return [gr.Column(visible=False), gr.Row(visible=False), gr.Row(visible=False)]
    elif dataset_type=="local":
        return [gr.Column(visible=True), gr.Row(visible=True), gr.Row(visible=False)]
    return [gr.Column(visible=True), gr.Row(visible=False), gr.Row(visible=True)]
  
def pgd_show_label_output(dataset_type):
    '''
    Show PGD output component based on dataset type
    '''
    if dataset_type=="local":
        return [gr.Label(visible=True), gr.Label(visible=True), gr.Number(visible=False), gr.Number(visible=False), gr.Number(visible=True)]
    return [gr.Label(visible=False), gr.Label(visible=False), gr.Number(visible=True), gr.Number(visible=True), gr.Number(visible=True)]

def pgd_update_epsilon(clip_values):
    '''
    Update max value of PGD epsilon slider based on model clip values
    '''
    if clip_values == 255:
        return gr.Slider(minimum=0.0001, maximum=255, label="Epslion", value=55) 
    return gr.Slider(minimum=0.0001, maximum=1, label="Epslion", value=0.05) 

def patch_show_label_output(dataset_type):
    '''
    Show adversarial patch output components based on dataset type
    '''
    if dataset_type=="local":
        return [gr.Label(visible=True), gr.Label(visible=True), gr.Number(visible=False), gr.Number(visible=False), gr.Number(visible=True)]
    return [gr.Label(visible=False), gr.Label(visible=False), gr.Number(visible=True), gr.Number(visible=True), gr.Number(visible=True)]

def show_target_label_dataframe(dataset_type):
    if dataset_type == "Example CIFAR10":
        return gr.Dataframe(visible=True), gr.Dataframe(visible=False)
    elif dataset_type == "Example XView":
        return gr.Dataframe(visible=False), gr.Dataframe(visible=True)
    return gr.Dataframe(visible=False), gr.Dataframe(visible=False)
    
# e.g. To use a local alternative theme: carbon_theme = Carbon()
with gr.Blocks(css=css, theme='xiaobaiyuan/theme_brief') as demo:
    gr.Markdown("<h1>HEART Adversarial Robustness Gradio Example</h1>")
    
    with gr.Tab("Classification", elem_classes="task-tab"):
        gr.Markdown("Classifying images with a set of categories.")
        
        # Model and Dataset Selection
        with gr.Row():
            # Model and Dataset type e.g. Torchvision, HuggingFace, local etc.
            with gr.Column():
                model_type = gr.Radio(label="Model type", choices=["Example CIFAR10", "Example XView"],
                                    value="Example CIFAR10")
                dataset_type = gr.Radio(label="Dataset", choices=["Example CIFAR10", "Example XView"],
                                    value="Example CIFAR10")
            # Model parameters e.g. RESNET, VIT, input dimensions, clipping values etc.
            with gr.Column(visible=False) as model_params:
                model_path = gr.Textbox(placeholder="URL", label="Model path")
                with gr.Row():
                    with gr.Column():
                        model_channels = gr.Textbox(placeholder="Integer, 3 for RGB images", label="Input Channels", value=3)
                    with gr.Column():
                        model_width = gr.Textbox(placeholder="Integer", label="Input Width", value=640)
                with gr.Row():
                    with gr.Column():
                        model_height = gr.Textbox(placeholder="Integer", label="Input Height", value=480)
                    with gr.Column():
                        model_clip = gr.Radio(choices=[1, 255], label="Pixel clip", value=1)
            # Dataset parameters e.g. Torchvision, HuggingFace, local etc. 
            with gr.Column(visible=False) as dataset_params:
                with gr.Row() as local_image:
                    image = gr.Image(sources=['upload'], type="pil", height=150, width=150, elem_classes="input-image")
                with gr.Row() as hosted_image:
                    dataset_path = gr.Textbox(placeholder="URL", label="Dataset path")
                    dataset_split = gr.Textbox(placeholder="test", label="Dataset split")
            
            model_type.change(show_model_params, model_type, model_params)
            dataset_type.change(show_dataset_params, dataset_type, [dataset_params, local_image, hosted_image])
        
        # Attack Selection
        with gr.Row():
            
            with gr.Tab("Info"):
                gr.Markdown("This is step 1. Select the type of attack for evaluation.")
                
            with gr.Tab("White Box"):
                gr.Markdown("White box attacks assume the attacker has __full access__ to the model.")
                
                with gr.Tab("Info"):
                    gr.Markdown("This is step 2. Select the type of white-box attack to evaluate.")
                
                with gr.Tab("Evasion"):
                    gr.Markdown("Evasion attacks are deployed to cause a model to incorrectly classify or detect items/objects in an image.")
                    
                    with gr.Tab("Info"):
                        gr.Markdown("This is step 3. Select the type of Evasion attack to evaluate.")
                    
                    with gr.Tab("Projected Gradient Descent"):
                        gr.Markdown("This attack uses PGD to identify adversarial examples.")
                        
                        
                        with gr.Row():
                            
                            with gr.Column(scale=1):
                                attack = gr.Textbox(visible=True, value="PGD", label="Attack", interactive=False)
                                max_iter = gr.Slider(minimum=1, maximum=20, label="Max iterations", value=10, step=1)
                                eps = gr.Slider(minimum=2, maximum=255, label="Epslion", value=8, step=1) 
                                eps_steps = gr.Slider(minimum=1, maximum=254, label="Epsilon steps", value=2, step=1) 
                                targeted = gr.Textbox(placeholder="Target label (integer)", label="Target")
                                with gr.Accordion("Target mapping", open=False):
                                    cifar_labels = gr.Dataframe(pd.DataFrame(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                                                                columns=['label']).rename_axis('target').reset_index(),
                                                                visible=True, elem_classes=["small-font", "df-padding"],
                                                                type="pandas",interactive=False)
                                    xview_labels = gr.Dataframe(pd.DataFrame(['Building', 'Construction Site', 'Engineering Vehicle', 'Fishing Vessel', 'Oil Tanker', 
                                                                            'Vehicle Lot'],
                                                                columns=['label']).rename_axis('target').reset_index(), 
                                                                visible=False, elem_classes=["small-font", "df-padding"],
                                                                type="pandas",interactive=False)
                                    
                                eval_btn_pgd = gr.Button("Evaluate")
                                model_clip.change(pgd_update_epsilon, model_clip, eps)
                                dataset_type.change(show_target_label_dataframe, dataset_type, [cifar_labels, xview_labels])
                                
                            # Evaluation Output. Visualisations of success/failures of running evaluation attacks.
                            with gr.Column(scale=2):
                                with gr.Row():
                                    with gr.Column():
                                        original_gallery = gr.Gallery(label="Original", preview=True, height=600)
                                        benign_output = gr.Label(num_top_classes=3, visible=False)
                                        clean_accuracy = gr.Number(label="Clean Accuracy", precision=2)
                                        
                                    with gr.Column():
                                        adversarial_gallery = gr.Gallery(label="Adversarial", preview=True, height=600)
                                        adversarial_output = gr.Label(num_top_classes=3, visible=False)
                                        robust_accuracy = gr.Number(label="Robust Accuracy", precision=2)
                                        perturbation_added = gr.Number(label="Perturbation Added", precision=2)
                                        
                                dataset_type.change(pgd_show_label_output, dataset_type, [benign_output, adversarial_output, 
                                                                                     clean_accuracy, robust_accuracy, perturbation_added])
                                eval_btn_pgd.click(clf_evasion_evaluate, inputs=[attack, model_type, model_path, model_channels, model_height, model_width,
                                                                             model_clip, max_iter, eps, eps_steps, targeted, 
                                                                             dataset_type, dataset_path, dataset_split, image],
                                                    outputs=[original_gallery, benign_output, adversarial_gallery, adversarial_output, clean_accuracy,
                                                             robust_accuracy, perturbation_added], api_name='patch')
                        
                        with gr.Row():
                            clear_btn = gr.ClearButton([image, targeted, original_gallery, benign_output, clean_accuracy,
                                                        adversarial_gallery, adversarial_output, robust_accuracy, perturbation_added])
                            
                    with gr.Tab("Adversarial Patch"):
                        gr.Markdown("This attack crafts an adversarial patch that facilitates evasion.")
                        
                        with gr.Row():
                            
                            with gr.Column(scale=1):
                                with gr.Accordion('Adversarial Patch Parameters', open=False):
                                    attack = gr.Textbox(visible=True, value="Adversarial Patch", label="Attack", interactive=False)
                                    max_iter = gr.Slider(minimum=1, maximum=20, label="Max iterations", value=2, step=1)
                                    patch_dim = gr.Slider(minimum=1, maximum=32, label="Patch dimension", value=6, step=1, info="The height and width of the patch") 
                                    x_location = gr.Slider(minimum=0, maximum=25, label="Location (x)", value=1, step=1, info="Shift patch left and right") 
                                    y_location = gr.Slider(minimum=0, maximum=25, label="Location (y)", value=1, step=1, info="Shift patch up and down") 
                                    targeted = gr.Textbox(placeholder="Target label (integer)", label="Target")
                                    
                                    dataset_type.change(update_patch_sliders, 
                                                      [x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image],
                                                      [patch_dim, x_location, y_location])
                                    image.change(update_patch_sliders, 
                                                      [x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image],
                                                      [patch_dim, x_location, y_location])
                                    patch_dim.release(update_patch_sliders, 
                                                      [x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image],
                                                      [patch_dim, x_location, y_location])
                            
                            with gr.Column(scale=1):
                                #adding in preview option for patch location
                                with gr.Accordion('Preview Patch Placement', open=False):
                                    gr.Markdown('''<i>Using the location (x and y) and patch size (height and width) controls in the <b>parameters</b> 
                                                section, you can control how the adversarial patch is positioned.</i>''')
                                    with gr.Column():
                                        test_patch_gallery = gr.Image(show_label=False, show_download_button=False, elem_classes="output-image")
                                
                                    preview_patch_loc = gr.Button('Preview Patch Placement')
                                    preview_patch_loc.click(preview_patch_location, inputs=[x_location, y_location, patch_dim,
                                                                                                dataset_type, dataset_path, dataset_split, image],
                                                                outputs = [test_patch_gallery])
                            with gr.Column(scale=1):
                                with gr.Accordion('Target Mapping', open=False):
                                    gr.Markdown('''<i>If deploying a targeted attack, use the mapping of classes
                                                to integer below to populate the <b>target label</b> box in the parameters section.</i>''')
                                    cifar_labels = gr.Dataframe(pd.DataFrame(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
                                                                columns=['label']).rename_axis('target').reset_index(),
                                                                visible=True, elem_classes=["small-font", "df-padding"],
                                                                type="pandas",interactive=False)
                                    xview_labels = gr.Dataframe(pd.DataFrame(['Building', 'Construction Site', 'Engineering Vehicle', 'Fishing Vessel', 'Oil Tanker', 
                                                                            'Vehicle Lot'],
                                                                columns=['label']).rename_axis('target').reset_index(), 
                                                                visible=False, elem_classes=["small-font", "df-padding"],
                                                                type="pandas",interactive=False)
                        with gr.Row():
                            eval_btn_patch = gr.Button("Evaluate")
                            dataset_type.change(show_target_label_dataframe, dataset_type, [cifar_labels, xview_labels])
                        with gr.Row():        
                            # Evaluation Output. Visualisations of success/failures of running evaluation attacks.
                            
                            with gr.Column(scale=2):
                                original_gallery = gr.Gallery(label="Original", preview=True, height=600)
                                benign_output = gr.Label(num_top_classes=3, visible=False)
                                clean_accuracy = gr.Number(label="Clean Accuracy", precision=2)
                                
                            with gr.Column(scale=2):
                                adversarial_gallery = gr.Gallery(label="Adversarial", preview=True, height=600)
                                adversarial_output = gr.Label(num_top_classes=3, visible=False)
                                robust_accuracy = gr.Number(label="Robust Accuracy", precision=2)
                            with gr.Column(scale=1):
                                patch_image = gr.Image(label="Adversarial Patch")
                                    
                            dataset_type.change(patch_show_label_output, dataset_type, [benign_output, adversarial_output, 
                                                                                    clean_accuracy, robust_accuracy, patch_image])
                            eval_btn_patch.click(clf_evasion_evaluate, inputs=[attack, model_type, model_path, model_channels, model_height, model_width,
                                                                            model_clip, max_iter, x_location, y_location, patch_dim, targeted, 
                                                                            dataset_type, dataset_path, dataset_split, image],
                                                outputs=[original_gallery, benign_output, adversarial_gallery, adversarial_output, clean_accuracy,
                                                            robust_accuracy, patch_image])
                        
                        with gr.Row():
                            clear_btn = gr.ClearButton([image, targeted, original_gallery, benign_output, clean_accuracy,
                                                        adversarial_gallery, adversarial_output, robust_accuracy, patch_image])
                        
                with gr.Tab("Poisoning"):
                    gr.Markdown("Coming soon.")
            
            with gr.Tab("Black Box"):
                gr.Markdown("Black box attacks assume the attacker __does not__ have full access to the model but can query it for predictions.")
                
                with gr.Tab("Info"):
                    gr.Markdown("This is step 2. Select the type of black-box attack to evaluate.")
                    
                with gr.Tab("Evasion"):
                    
                    gr.Markdown("Evasion attacks are deployed to cause a model to incorrectly classify or detect items/objects in an image.")
                    
                    with gr.Tab("Info"):
                        gr.Markdown("This is step 3. Select the type of Evasion attack to evaluate.")
                    
                    with gr.Tab("HopSkipJump"):
                        gr.Markdown("Coming soon.")
                    
                    with gr.Tab("Square Attack"):
                        gr.Markdown("Coming soon.")
                        
            with gr.Tab("AutoAttack"):
                gr.Markdown("Coming soon.")
            
            
    with gr.Tab("Object Detection"):
        gr.Markdown("Extracting objects from images and identifying their category.")
        
        # Model and Dataset Selection
        with gr.Row():
            # Model and Dataset type e.g. Torchvision, HuggingFace, local etc.
            with gr.Column():
                model_type = gr.Radio(label="Model type", choices=["DeTR"],
                                    value="DeTR")
                dataset_type = gr.Radio(label="Dataset", choices=["COCO", "VisDrone"],
                                    value="COCO")
            
            model_type.change(show_model_params, model_type, model_params)
            dataset_type.change(show_dataset_params, dataset_type, [dataset_params, local_image])
        
        # Attack Selection
        with gr.Row():
            
            with gr.Tab("Info"):
                gr.Markdown("This is step 1. Select the type of attack for evaluation.")
                
            with gr.Tab("White Box"):
                gr.Markdown("White box attacks assume the attacker has __full access__ to the model.")
                
                with gr.Tab("Info"):
                    gr.Markdown("This is step 2. Select the type of white-box attack to evaluate.")
                
                with gr.Tab("Evasion"):
                    gr.Markdown("Evasion attacks are deployed to cause a model to incorrectly classify or detect items/objects in an image.")
                    
                    with gr.Tab("Info"):
                        gr.Markdown("This is step 3. Select the type of Evasion attack to evaluate.")
                    
                    with gr.Tab("Projected Gradient Descent"):
                        gr.Markdown("This attack uses PGD to identify adversarial examples.")
                        
                        
                        with gr.Row():
                            
                            with gr.Column(scale=1):
                                attack = gr.Textbox(visible=True, value="PGD", label="Attack", interactive=False)
                                max_iter = gr.Slider(minimum=1, maximum=10, label="Max iterations", value=2, step=1)
                                eps = gr.Slider(minimum=0.01, maximum=1, label="Epslion", value=0.01, step=0.01)
                                eps_steps = gr.Slider(minimum=0.001, maximum=0.9, label="Epsilon steps", value=0.001, step=0.001)
                                targeted = gr.Textbox(placeholder="Target label (integer)", label="Target")
                                det_threshold = gr.Slider(minimum=0.0, maximum=1.0, label="Detection threshold", value=0.7, step=0.05)
                                eval_btn_pgd = gr.Button("Evaluate")
                                model_clip.change(pgd_update_epsilon, model_clip, eps)
                                
                            # Evaluation Output. Visualisations of success/failures of running evaluation attacks.
                            with gr.Column(scale=3):
                                with gr.Row():
                                    with gr.Column():
                                        original_gallery = gr.Gallery(label="Original", preview=True, show_download_button=True, height=600)
                                        clean_map = gr.Number(label="Clean mAP@50", precision=2)
                                        
                                    with gr.Column():
                                        adversarial_gallery = gr.Gallery(label="Adversarial", preview=True, show_download_button=True, height=600)
                                        robust_map = gr.Number(label="Robust mAP@50", precision=2)
                                        perturbation_added = gr.Number(label="Perturbation Added", precision=2)
                                        
                                eval_btn_pgd.click(det_evasion_evaluate, inputs=[attack, model_type, model_path, model_channels, model_height, model_width,
                                                                             model_clip, max_iter, eps, eps_steps, targeted, 
                                                                             det_threshold, dataset_type, image],
                                                    outputs=[original_gallery, adversarial_gallery, clean_map, robust_map, perturbation_added], api_name='patch')
                        
                        with gr.Row():
                            clear_btn = gr.ClearButton([image, original_gallery,
                                                        adversarial_gallery, clean_map, robust_map, perturbation_added])
                    with gr.Tab("Adversarial Patch"):
                        gr.Markdown("This attack crafts an adversarial patch that facilitates evasion.")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Accordion("Adversarial Patch Parameters", open=False):
                                    attack = gr.Textbox(visible=True, value="Adversarial Patch", label="Attack", interactive=False)
                                    max_iter = gr.Slider(minimum=1, maximum=100, label="Max iterations", value=1, step=1)
                                    patch_dim = gr.Slider(minimum=1, maximum=640, label="Patch dimension", value=100, step=1, info="The height and width of the patch") 
                                    x_location = gr.Slider(minimum=0, maximum=640, label="Location (x)", value=100, step=1, info="Shift patch left and right") 
                                    y_location = gr.Slider(minimum=0, maximum=480, label="Location (y)", value=100, step=1, info="Shift patch up and down") 
                                    targeted = gr.Textbox(placeholder="Target label (integer)", label="Target")
                                    det_threshold = gr.Slider(minimum=0.0, maximum=1.0, label="Detection threshold", value=0.7, step=0.05)
                                    
                                    dataset_type.change(update_patch_sliders, 
                                                      [x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image],
                                                      [patch_dim, x_location, y_location])
                                    image.change(update_patch_sliders, 
                                                      [x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image],
                                                      [patch_dim, x_location, y_location])
                                    patch_dim.release(update_patch_sliders, 
                                                      [x_location, y_location, patch_dim, dataset_type, dataset_path, dataset_split, image],
                                                      [patch_dim, x_location, y_location])
                            
                            with gr.Column(scale=1):
                                #adding in preview option for patch location
                                with gr.Accordion('Preview Patch Placement', open=False):
                                    gr.Markdown('''<i>Using the location (x and y) and patch size (height and width) controls in the <b>parameters</b> 
                                                section, you can control how the adversarial patch is positioned.</i>''')
                                    with gr.Column():
                                        test_patch_gallery = gr.Image(show_label=False, show_download_button=False, width=300, height=300, elem_classes=["output-image"])
                                
                                    preview_patch_loc = gr.Button('Preview Patch Placement')
                                    preview_patch_loc.click(preview_patch_location, inputs=[x_location, y_location, patch_dim,
                                                                                                dataset_type, dataset_path, dataset_split, image],
                                                                outputs = [test_patch_gallery])
                            
                            with gr.Column(scale=1):
                                with gr.Accordion('Target Mapping', open=False):
                                    gr.Markdown('''<i>If deploying a targeted attack, use the mapping of classes
                                                to integer below to populate the <b>target label</b> box in the parameters section.</i>''')
                                    coco_labels = gr.Dataframe(pd.DataFrame(['N/A','person','bicycle','car','motorcycle','airplane','bus','train',
                                                                            'truck','boat','traffic light','fire hydrant','N/A','stop sign','parking meter','bench',
                                                                            'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','N/A',
                                                                            'backpack','umbrella','N/A','N/A','handbag','tie','suitcase','frisbee','skis','snowboard',
                                                                            'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
                                                                            'bottle','N/A','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                                                                            'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
                                                                            'bed','N/A','dining table','N/A','N/A','toilet','N/A','tv','laptop','mouse','remote','keyboard',
                                                                            'cell phone','microwave','oven','toaster','sink','refrigerator','N/A','book','clock',
                                                                            'vase','scissors','teddy bear','hair drier','toothbrush'],
                                                                columns=['label']).rename_axis('target').reset_index(),
                                                                visible=True, elem_classes=["small-font", "df-padding"],
                                                                type="pandas",interactive=False)
                        
                        with gr.Row():
                            eval_btn_patch = gr.Button("Evaluate")
                                
                        with gr.Row():
                            # Evaluation Output. Visualisations of success/failures of running evaluation attacks.
                            with gr.Column(scale=3):
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        original_gallery = gr.Gallery(label="Original", preview=True, show_download_button=True, height=600)
                                        clean_map = gr.Number(label="Clean mAP@50", precision=2)
                                        
                                    with gr.Column(scale=2):
                                        adversarial_gallery = gr.Gallery(label="Adversarial", preview=True, show_download_button=True, height=600)
                                        robust_map = gr.Number(label="Robust mAP@50", precision=2)
                                   
                                    with gr.Column(scale=1):     
                                        patch_image = gr.Image(label="Adversarial Patch")
                                        
                                dataset_type.change(patch_show_label_output, dataset_type, [adversarial_output, ])
                                eval_btn_patch.click(det_evasion_evaluate, inputs=[attack, model_type, model_path, model_channels, model_height, model_width,
                                                                             model_clip, max_iter, x_location, y_location, patch_dim, targeted, 
                                                                             det_threshold,dataset_type, image],
                                                    outputs=[original_gallery, adversarial_gallery, clean_map, robust_map, patch_image])
                        
                        with gr.Row():
                            clear_btn = gr.ClearButton([image, targeted, original_gallery,
                                                        adversarial_gallery, clean_map, robust_map])
                        
                with gr.Tab("Poisoning"):
                    gr.Markdown("Coming soon.")
            
            with gr.Tab("Black Box"):
                gr.Markdown("Black box attacks assume the attacker __does not__ have full access to the model but can query it for predictions.")
                
                with gr.Tab("Info"):
                    gr.Markdown("This is step 2. Select the type of black-box attack to evaluate.")
                    
                with gr.Tab("Evasion"):
                    
                    gr.Markdown("Evasion attacks are deployed to cause a model to incorrectly classify or detect items/objects in an image.")
                    
                    with gr.Tab("Info"):
                        gr.Markdown("This is step 3. Select the type of Evasion attack to evaluate.")
                    
                    with gr.Tab("HopSkipJump"):
                        gr.Markdown("Coming soon.")
                    
                    with gr.Tab("Square Attack"):
                        gr.Markdown("Coming soon.")
                        
            with gr.Tab("AutoAttack"):
                gr.Markdown("Coming soon.")


def launch_demo_via_huggingface():
    """
    Hardened Extension of Adversarial Robustness Toolbox (HEART) has not yet been opensourced to Pypi.
    Until this is completed, the HEART library must be installed via a private repository.
    This launch method gets private secretes from Huggingface and executes HEART install via pip.
    
    TODO [HEART Issue#13]: Tear down this Huggingface demo launch switch once HEART has been fully opensourced.
    """

    import os, re
    from pip._internal.cli.main import main as pipmain
    
    # Huggingface does not support LFS via external https, disable smudge
    os.putenv('GIT_LFS_SKIP_SMUDGE', '1')

    # Get protected private repository installation command from Huggingface secrets
    HEART_INSTALL=os.environ['HEART_INSTALL']
    HEART_REGEX=r"git\+https\:\/\/[a-zA-Z]{9}\:[a-zA-Z0-9\-\_]{26}\@gitlab\.jatic\.net\/jatic\/ibm\/hardened-extension-adversarial-robustness-toolbox\.git"

    # Execute pip install
    if re.match(HEART_REGEX, HEART_INSTALL):
        pipmain(['install', HEART_INSTALL])
    else:
        print(
            f"""
            The HEART library was not installed. Credentials supplied were most likely incorrect.
            Install string supplied did not match filter: {HEART_REGEX}
            """
        )
    
    demo.launch()


def launch_demo_via_local():
    """
    Default functionality of launching the Gradio app from any local development environment.
    This launch method assumes that the local environment can launch a web browser from within the
    same local environment and navigate to the local host shown in demo.launch() output.

    * Important Notes:
      - This launch mechanism will not function via Huggingface.
      - When launching via Raven, share must be set to True (Raven has no local web browser).
    """

    # during development, set debug=True
    demo.launch(show_api=False, debug=True, share=False,
                server_name="0.0.0.0", 
                server_port=7777, 
                ssl_verify=False,
                max_threads=20)


if __name__ == "__main__":

    import socket

    # Huggingface Hostname Patterns 
    HF_SPACES=[
        "alpha-heart-gradio", 
        "cdao-heart-gradio",
    ]

    # Try to describe hostname using socket. If this doesn't work, fail open as local.  
    hostname = ""

    try:
        print(f"Attempting to resolve hostname via socket.gethostname()...")
        hostname = socket.gethostname()
        print(f"Hostname resolved successfully as: {hostname}")
    except:
        print(f"Unable to resolve hostname via socket.gethostname()...")
        hostname = "local"
        print(f"Defaulting to hostname set as: local")

    if any(space in hostname for space in HF_SPACES):
        print(
            f"""
            [{hostname}] is most likely within a Huggingface Space.
            Current understood list of HF_SPACES: {HF_SPACES}
            Executing demo.launch() using <launch_demo_via_huggingface()>
            """
        )
        launch_demo_via_huggingface()
    else:
        print(
            f"""
            {hostname} is either local or uncaptured for demo.launch() switching.
            Executing demo.launch() using <launch_demo_via_local()>
            """
        )
        launch_demo_via_local()
