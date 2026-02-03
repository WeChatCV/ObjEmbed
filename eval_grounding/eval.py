import os
import time
import json
import random
import argparse
import itertools
import subprocess
import torch
import torch.nn.functional as F
import copy
import torchvision
from torchvision.ops import batched_nms
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from d_cube import D3
from lvis import LVIS, LVISEval, LVISResults
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

from recall import bbox_overlaps, eval_recalls



def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']
        
def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']
        

def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary_uncleaned = [get_category_name(id, categories) for id in vocabulary_id]
    return vocabulary_uncleaned, vocabulary_id


ds_collections = {
    'coco': {
        'ann_path': 'datasets/coco/annotations/instances_val2017.json',
        'task_specific_visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'img_path': 'datasets/coco/val2017/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/coco_proposals_all.json',
        'classes_en': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    },
    'lvis': {
        'ann_path': 'datasets/coco/lvis/lvis_v1_val.json',
        'task_specific_visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'img_path': 'datasets/coco/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/lvis_v1_proposals_all.json',
    },
    'coco_o': {
        'ann_path': [
            'datasets/ood_coco/cartoon/annotations/instances_val2017.json',
            'datasets/ood_coco/handmake/annotations/instances_val2017.json',
            'datasets/ood_coco/painting/annotations/instances_val2017.json',
            'datasets/ood_coco/sketch/annotations/instances_val2017.json',
            'datasets/ood_coco/tattoo/annotations/instances_val2017.json',
            'datasets/ood_coco/weather/annotations/instances_val2017.json'
        ],
        'subsets': ['cartoon', 'handmake', 'painting', 'sketch', 'tattoo', 'weather'],
        'task_specific_visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'img_path': 'datasets/ood_coco/%s/val2017/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/coco_o_proposals_all.json',
    },
    'refcoco': {
        'ann_path': [
            'datasets/refcoco/refcoco_validation.json',
            'datasets/refcoco/refcoco_test.json',
            'datasets/refcoco/refcoco_testB.json',
            'datasets/refcocoplus/refcocoplus_validation.json',
            'datasets/refcocoplus/refcocoplus_test.json',
            'datasets/refcocoplus/refcocoplus_testB.json',
            'datasets/refcocog/refcocog_validation.json',
            'datasets/refcocog/refcocog_test.json',
        ],
        'task_specific_visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'img_path': 'datasets/coco2014/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/refcoco_proposals_all.json',
    },
    'd3': {
        'ann_path': [
            'datasets/d3/d3_json/d3_full_annotations.json',
            'datasets/d3/d3_json/d3_pres_annotations.json',
            'datasets/d3/d3_json/d3_abs_annotations.json',
        ],
        'task_specific_visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'img_path': 'datasets/d3/d3_images/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/d3_proposals_all.json',
    },
    'odinw35': {
        'prposals': 'datasets/wedetect_ref/eval_proposals/odinw35_proposals_all.json',
        'task_specific_visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'datasets': {
            # 1
            'AerialMaritimeDrone_large': {
                'ann_path': 'datasets/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/',
                'classes_en': ['boat', 'car', 'dock', 'jetski', 'lift'],
            },
            # 2
            'AerialMaritimeDrone_tiled': {
                'ann_path': 'datasets/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/tiled/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/tiled/valid/',
                'classes_en': ['boat', 'car', 'dock', 'jetski', 'lift'],
            },
            # 3
            'AmericanSignLanguageLetters': {
                'ann_path': 'datasets/ODinW35/AmericanSignLanguageLetters/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/AmericanSignLanguageLetters/AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/valid/',
                'classes_en': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            },
            # 4
            'Aquarium': {
                'ann_path': 'datasets/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/',
                'classes_en': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'],
            },
            # 5
            'BCCD': {
                'ann_path': 'datasets/ODinW35/BCCD/BCCD/BCCD.v3-raw.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/BCCD/BCCD/BCCD.v3-raw.coco/valid/',
                'classes_en': ['Platelets', 'RBC', 'WBC'],
            },
            # 6
            'boggleBoards': {
                'ann_path': 'datasets/ODinW35/boggleBoards/boggleBoards/416x416AutoOrient/export/val_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/boggleBoards/boggleBoards/416x416AutoOrient/export/',
                'classes_en': ['Q', 'a', 'an', 'b', 'c', 'd', 'e', 'er', 'f', 'g', 'h', 'he', 'i', 'in', 'j', 'k', 'l', 'm', 'n', 'o', 'o ', 'p', 'q', 'qu', 'r', 's', 't', 't\\', 'th', 'u', 'v', 'w', 'wild', 'x', 'y', 'z'],
            },
            # 7
            'brackishUnderwater': {
                'ann_path': 'datasets/ODinW35/brackishUnderwater/brackishUnderwater/960x540/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/brackishUnderwater/brackishUnderwater/960x540/valid/',
                'classes_en': ['crab', 'fish', 'jellyfish', 'shrimp', 'small_fish', 'starfish'],
            },
            # 8
            'ChessPieces': {
                'ann_path': 'datasets/ODinW35/ChessPieces/ChessPieces/Chess_Pieces.v23-raw.coco/valid/new_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/ChessPieces/ChessPieces/Chess_Pieces.v23-raw.coco/valid/',
                'classes_en': ['  ', 'black bishop', 'black king', 'black knight', 'black pawn', 'black queen', 'black rook', 'white bishop', 'white king', 'white knight', 'white pawn', 'white queen', 'white rook'],
            },
            # 9
            'CottontailRabbits': {
                'ann_path': 'datasets/ODinW35/CottontailRabbits/CottontailRabbits/valid/new_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/CottontailRabbits/CottontailRabbits/valid/',
                'classes_en': ['rabbit'],
            },
            # 10
            'dice': {
                'ann_path': 'datasets/ODinW35/dice/dice/mediumColor/export/val_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/dice/dice/mediumColor/export/',
                'classes_en': ['1', '2', '3', '4', '5', '6'],
            },
            # 11
            'DroneControl': {
                'ann_path': 'datasets/ODinW35/DroneControl/DroneControl/Drone Control.v3-raw.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/DroneControl/DroneControl/Drone Control.v3-raw.coco/valid/',
                'classes_en': ['follow', 'follow_hand', 'land', 'land_hand', 'null', 'object', 'takeoff', 'takeoff-hand'],
            },
            # 12
            'EgoHands_generic': {
                'ann_path': 'datasets/ODinW35/EgoHands/EgoHands/generic/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/EgoHands/EgoHands/generic/valid/',
                'classes_en': ['hand'],
            },
            # 13
            'EgoHands_specific': {
                'ann_path': 'datasets/ODinW35/EgoHands/EgoHands/specific/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/EgoHands/EgoHands/specific/valid/',
                'classes_en': ['myleft', 'myright', 'yourleft', 'yourright'],
            },
            # 14
            'HardHatWorkers': {
                'ann_path': 'datasets/ODinW35/HardHatWorkers/HardHatWorkers/raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/HardHatWorkers/HardHatWorkers/raw/valid/',
                'classes_en': ['head', 'helmet', 'person'],
            },
            # 15
            'MaskWearing': {
                'ann_path': 'datasets/ODinW35/MaskWearing/MaskWearing/raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/MaskWearing/MaskWearing/raw/valid/',
                'classes_en': ['mask', 'no-mask'],
            },
            # 16
            'MountainDewCommercial': {
                'ann_path': 'datasets/ODinW35/MountainDewCommercial/MountainDewCommercial/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/MountainDewCommercial/MountainDewCommercial/valid/',
                'classes_en': ['bottle'],
            },
            # 17
            'NorthAmericaMushrooms': {
                'ann_path': 'datasets/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/new_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/',
                'classes_en': ['flat mushroom', 'yellow mushroom'],
            },
            # 18
            'openPoetryVision': {
                'ann_path': 'datasets/ODinW35/openPoetryVision/openPoetryVision/512x512/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/openPoetryVision/openPoetryVision/512x512/valid/',
                'classes_en': ['American Typewriter', 'Andale Mono', 'Apple Chancery', 'Arial', 'Avenir', 'Baskerville', 'Big Caslon', 'Bradley Hand', 'Brush Script MT', 'Chalkboard', 'Comic Sans MS', 'Copperplate', 'Courier', 'Didot', 'Futura', 'Geneva', 'Georgia', 'Gill Sans', 'Helvetica', 'Herculanum', 'Impact', 'Kefa', 'Lucida Grande', 'Luminari', 'Marker Felt', 'Menlo', 'Monaco', 'Noteworthy', 'Optima', 'PT Sans', 'PT Serif', 'Palatino', 'Papyrus', 'Phosphate', 'Rockwell', 'SF Pro', 'SignPainter', 'Skia', 'Snell Roundhand', 'Tahoma', 'Times New Roman', 'Trebuchet MS', 'Verdana'],
            },
            # 19
            'OxfordPets_by_breed': {
                'ann_path': 'datasets/ODinW35/OxfordPets/OxfordPets/by-breed/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/OxfordPets/OxfordPets/by-breed/valid/',
                'classes_en': ['cat-Abyssinian', 'cat-Bengal', 'cat-Birman', 'cat-Bombay', 'cat-British_Shorthair', 'cat-Egyptian_Mau', 'cat-Maine_Coon', 'cat-Persian', 'cat-Ragdoll', 'cat-Russian_Blue', 'cat-Siamese', 'cat-Sphynx', 'dog-american_bulldog', 'dog-american_pit_bull_terrier', 'dog-basset_hound', 'dog-beagle', 'dog-boxer', 'dog-chihuahua', 'dog-english_cocker_spaniel', 'dog-english_setter', 'dog-german_shorthaired', 'dog-great_pyrenees', 'dog-havanese', 'dog-japanese_chin', 'dog-keeshond', 'dog-leonberger', 'dog-miniature_pinscher', 'dog-newfoundland', 'dog-pomeranian', 'dog-pug', 'dog-saint_bernard', 'dog-samoyed', 'dog-scottish_terrier', 'dog-shiba_inu', 'dog-staffordshire_bull_terrier', 'dog-wheaten_terrier', 'dog-yorkshire_terrier'],
            },
            # 20
            'OxfordPets_by_species': {
                'ann_path': 'datasets/ODinW35/OxfordPets/OxfordPets/by-species/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/OxfordPets/OxfordPets/by-species/valid/',
                'classes_en': ['cat', 'dog'],
            },
            # 21
            'PKLot': {
                'ann_path': 'datasets/ODinW35/PKLot/PKLot/640/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/PKLot/PKLot/640/valid/',
                'classes_en': ['space-empty', 'space-occupied'],
            },
            # 22
            'Packages': {
                'ann_path': 'datasets/ODinW35/Packages/Packages/Raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/Packages/Packages/Raw/valid/',
                'classes_en': ['package'],
            },
            # 23
            'PascalVOC': {
                'ann_path': 'datasets/ODinW35/PascalVOC/PascalVOC/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/PascalVOC/PascalVOC/valid/',
                'classes_en': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            },
            # 24
            'pistols': {
                'ann_path': 'datasets/ODinW35/pistols/pistols/export/val_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/pistols/pistols/export/',
                'classes_en': ['pistol'],
            },
            # 25
            'plantdoc': {
                'ann_path': 'datasets/ODinW35/plantdoc/plantdoc/416x416/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/plantdoc/plantdoc/416x416/valid/',
                'classes_en': ['Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf', 'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 'grape leaf', 'grape leaf black rot'],
            },
            # 26
            'pothole': {
                'ann_path': 'datasets/ODinW35/pothole/pothole/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/pothole/pothole/valid/',
                'classes_en': ['pothole'],
            },
            # 27
            'Raccoon': {
                'ann_path': 'datasets/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/',
                'classes_en': ['raccoon'],
            },
            # 28
            'selfdrivingCar': {
                'ann_path': 'datasets/ODinW35/selfdrivingCar/selfdrivingCar/fixedLarge/export/val_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/selfdrivingCar/selfdrivingCar/fixedLarge/export/',
                'classes_en': ['biker', 'car', 'pedestrian', 'trafficLight', 'trafficLight-Green', 'trafficLight-GreenLeft', 'trafficLight-Red', 'trafficLight-RedLeft', 'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck'],
            },
            # 29
            'ShellfishOpenImages': {
                'ann_path': 'datasets/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/',
                'classes_en': ['Crab', 'Lobster', 'Shrimp'],
            },
            # 30
            'ThermalCheetah': {
                'ann_path': 'datasets/ODinW35/ThermalCheetah/ThermalCheetah/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/ThermalCheetah/ThermalCheetah/valid/',
                'classes_en': ['cheetah', 'human'],
            },
            # 31
            'thermalDogsAndPeople': {
                'ann_path': 'datasets/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/',
                'classes_en': ['dog', 'person'],
            },
            # 32
            'UnoCards': {
                'ann_path': 'datasets/ODinW35/UnoCards/UnoCards/raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/UnoCards/UnoCards/raw/valid/',
                'classes_en': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            },
            # 33
            'VehiclesOpenImages': {
                'ann_path': 'datasets/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/',
                'classes_en': ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck'],
            },
            # 34
            'WildfireSmoke': {
                'ann_path': 'datasets/ODinW35/WildfireSmoke/WildfireSmoke/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/WildfireSmoke/WildfireSmoke/valid/',
                'classes_en': ['smoke'],
            },
            # 35
            'websiteScreenshots': {
                'ann_path': 'datasets/ODinW35/websiteScreenshots/websiteScreenshots/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/websiteScreenshots/websiteScreenshots/valid/',
                'classes_en': ['button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text'],
            },
        }
    },
    'odinw13': {
        'prposals': 'datasets/wedetect_ref/eval_proposals/odinw35_proposals_all.json',
        'task_specific_visual_prompt': 'Detect all objects in the image by identifying the common visual features of their respective classes. ',
        'datasets': {
            # 1
            'AerialMaritimeDrone_large': {
                'ann_path': 'datasets/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/AerialMaritimeDrone/AerialMaritimeDrone/large/valid/',
                'classes_en': ['boat', 'car', 'dock', 'jetski', 'lift'],
            },
            # 4
            'Aquarium': {
                'ann_path': 'datasets/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/Aquarium/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/',
                'classes_en': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'],
            },
            # 9
            'CottontailRabbits': {
                'ann_path': 'datasets/ODinW35/CottontailRabbits/CottontailRabbits/valid/new_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/CottontailRabbits/CottontailRabbits/valid/',
                'classes_en': ['rabbit'],
            },
            # 12
            'EgoHands_generic': {
                'ann_path': 'datasets/ODinW35/EgoHands/EgoHands/generic/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/EgoHands/EgoHands/generic/valid/',
                'classes_en': ['hand'],
            },
            # 17
            'NorthAmericaMushrooms': {
                'ann_path': 'datasets/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/new_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/NorthAmericaMushrooms/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/',
                'classes_en': ['flat mushroom', 'yellow mushroom'],
            },
            # 22
            'Packages': {
                'ann_path': 'datasets/ODinW35/Packages/Packages/Raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/Packages/Packages/Raw/valid/',
                'classes_en': ['package'],
            },
            # 23
            'PascalVOC': {
                'ann_path': 'datasets/ODinW35/PascalVOC/PascalVOC/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/PascalVOC/PascalVOC/valid/',
                'classes_en': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            },
            # 24
            'pistols': {
                'ann_path': 'datasets/ODinW35/pistols/pistols/export/val_annotations_without_background.json',
                'img_path': 'datasets/ODinW35/pistols/pistols/export/',
                'classes_en': ['pistol'],
            },
            # 26
            'pothole': {
                'ann_path': 'datasets/ODinW35/pothole/pothole/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/pothole/pothole/valid/',
                'classes_en': ['pothole'],
            },
            # 27
            'Raccoon': {
                'ann_path': 'datasets/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/Raccoon/Raccoon/Raccoon.v2-raw.coco/valid/',
                'classes_en': ['raccoon'],
            },
            # 29
            'ShellfishOpenImages': {
                'ann_path': 'datasets/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/ShellfishOpenImages/ShellfishOpenImages/raw/valid/',
                'classes_en': ['Crab', 'Lobster', 'Shrimp'],
            },
            # 31
            'thermalDogsAndPeople': {
                'ann_path': 'datasets/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/thermalDogsAndPeople/thermalDogsAndPeople/valid/',
                'classes_en': ['dog', 'person'],
            },
            # 33
            'VehiclesOpenImages': {
                'ann_path': 'datasets/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/annotations_without_background.json',
                'img_path': 'datasets/ODinW35/VehiclesOpenImages/VehiclesOpenImages/416x416/valid/',
                'classes_en': ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck'],
            },
        }
    },
    'FG-OVD': {
        'ann_path': [
            'datasets/FG-OVD/testset/1_attributes.json',
            'datasets/FG-OVD/testset/2_attributes.json',
            'datasets/FG-OVD/testset/3_attributes.json',
            'datasets/FG-OVD/testset/shuffle_negatives.json',
            'datasets/FG-OVD/testset/color.json',
            'datasets/FG-OVD/testset/material.json',
            'datasets/FG-OVD/testset/pattern.json',
            'datasets/FG-OVD/testset/transparency.json',
        ],
        'subset': ['Hard', 'Medium', 'Easy', 'Trivial', 'Color', 'Material', 'Pattern', 'Transparancy'],
        'n_hardnegatives': [5, 5, 5, 5, 2, 2, 2, 2],
        'task_specific_visual_prompt': 'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. ',
        'img_path': 'datasets/coco/',
        'proposals': 'datasets/wedetect_ref/eval_proposals/fg_ovd_proposals_all.json',
    },
}


class GroundingDataset(torch.utils.data.Dataset):
    

    def __init__(
        self,
        dataset: str,
        task_specific_visual_prompt=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.ann = []
        # self.query = ds_collections[dataset]['query'] if 'query' in ds_collections[dataset] else None
        self.task_specific_visual_prompt = task_specific_visual_prompt
        if self.task_specific_visual_prompt:
            visual_prompt = ds_collections[dataset]['task_specific_visual_prompt'] if 'task_specific_visual_prompt' in ds_collections[dataset] else ds_collections[dataset]['smoke']['task_specific_visual_prompt']
        else:
            visual_prompt = "Represent each objects and the whole image into a single token. "

        if dataset == 'coco':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            inverse_id_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
            images = json.load(open(ds_collections[dataset]['ann_path']))['images']
            coco = COCO(ds_collections[dataset]['ann_path'])
            for ann in images:
                gt_bboxes = []
                gt_labels = []
                ann_ids = coco.getAnnIds(imgIds=ann['id'])
                ann_infos = coco.loadAnns(ann_ids)
                for ann_info in ann_infos:
                    if ann_info.get('ignore', False) or ann_info['iscrowd']:
                        continue
                    x1, y1, w, h = ann_info['bbox']
                    gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                    gt_labels.append(str(inverse_id_map[ann_info['category_id']]))
                item = {
                    'id': ann['id'],
                    'image': ds_collections[dataset]['img_path'] + ann['file_name'],
                    'dataset': self.dataset,
                    'gt_labels': gt_labels,
                    'gt_bboxes': gt_bboxes,
                    'visual_prompt': visual_prompt,
                }
                self.ann.append(item)
            self.query = copy.deepcopy(ds_collections[dataset]['classes_en'])
        if dataset == 'lvis':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            lvis = LVIS(ds_collections[dataset]['ann_path'])
            # 获取所有类别名称
            cats = lvis.load_cats(lvis.get_cat_ids())
            self.query = [cat["name"] for cat in cats]
            img_ids = lvis.get_img_ids()
            for img_id in img_ids:
                gt_bboxes = []
                gt_labels = []
                img_info = lvis.load_imgs([img_id])[0]
                file_name = img_info["coco_url"][len('http://images.cocodataset.org/'):]
                image_path = ds_collections[dataset]['img_path'] + file_name
                ann_ids = lvis.get_ann_ids(img_ids=[img_id])
                anns = lvis.load_anns(ann_ids)
                for ann_info in anns:
                    if ann_info.get('ignore', False) or ann_info.get('iscrowd', False):
                        continue
                    x1, y1, w, h = ann_info['bbox']
                    gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                    gt_labels.append(str(ann_info['category_id'] - 1))
                item = {
                    'id': img_id,
                    'image': image_path,
                    'dataset': self.dataset,
                    'gt_labels': gt_labels,
                    'gt_bboxes': gt_bboxes,
                    'visual_prompt': visual_prompt,
                }
                self.ann.append(item)
        elif dataset == 'coco_o':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for ann_path, sub_dataset in zip(ds_collections[dataset]['ann_path'], ds_collections[dataset]['subsets']):
                coco = COCO(ann_path)
                img_ids = coco.getImgIds()
                for img_id in img_ids:
                    img_info = coco.loadImgs([img_id])[0]
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    ann_info = coco.loadAnns(ann_ids)
                    gt_bboxes = []
                    gt_labels = []
                    for ann in ann_info:
                        if ann.get('ignore', False) or ann.get('iscrowd', False):
                            continue
                        x1, y1, w, h = ann['bbox']
                        gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                        gt_labels.append(str(ann['category_id'] - 1))
                    item = {
                        'id': img_info['id'],
                        'image': ds_collections[dataset]['img_path'] % sub_dataset + img_info['file_name'],
                        'dataset': sub_dataset,
                        'gt_labels': gt_labels,
                        'gt_bboxes': gt_bboxes,
                        'visual_prompt': visual_prompt,
                    }
                    self.ann.append(item)
            self.query = copy.deepcopy(ds_collections['coco']['classes_en'])
        elif dataset == 'refcoco':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for ann_path in ds_collections[dataset]['ann_path']:
                data = json.load(open(ann_path))
                sub_dataset = ann_path.split('/')[-1].split('.')[0]
                for ann in data:
                    item = {
                        'id': ann['id'],
                        'image': ds_collections[dataset]['img_path'] + ann['image'],
                        'dataset': sub_dataset,
                        'query': [ann['conversations'][1]['value']],
                        'gt_labels': [ann['conversations'][1]['value']],
                        'gt_bboxes': ann['bounding_boxes'],
                        'visual_prompt': visual_prompt,
                    }
                    self.ann.append(item)
        elif dataset == 'd3':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for j, sub_dataset in enumerate(['FULL', 'PRES', 'ABS']):
                d3 = D3('datasets/d3/d3_images', 'datasets/d3/d3_pkl')
                img_ids = d3.get_img_ids()
                for i in range(len(img_ids)):
                    img_info = d3.load_imgs(img_ids[i])[0]
                    group_ids = d3.get_group_ids(img_ids=[img_ids[i]])
                    sent_ids = d3.get_sent_ids(group_ids=group_ids)
                    sent_list = d3.load_sents(sent_ids=sent_ids)
                    queries = [sent['raw_sent'] for sent in sent_list]
                    query_ids = [sent['id'] for sent in sent_list]
                    item = {
                        'id': img_info['id'],
                        'image': ds_collections[dataset]['img_path'] + img_info['file_name'],
                        'dataset': sub_dataset,
                        'query': queries,
                        'gt_labels': query_ids,
                        'gt_bboxes': [],
                        'visual_prompt': visual_prompt,
                    }
                    self.ann.append(item)
        elif dataset == 'odinw35' or dataset == 'odinw13':
            self.proposals = json.load(open(ds_collections[dataset]['prposals']))
            for sub_dataset_name, sub_dataset in ds_collections[dataset]['datasets'].items():
                images = json.load(open(sub_dataset['ann_path']))['images']
                coco = COCO(sub_dataset['ann_path'])
                for ann in images:
                    gt_bboxes = []
                    gt_labels = []
                    ann_ids = coco.getAnnIds(imgIds=ann['id'])
                    ann_infos = coco.loadAnns(ann_ids)
                    for ann_info in ann_infos:
                        if ann_info.get('ignore', False) or ann_info['iscrowd']:
                            continue
                        x1, y1, w, h = ann_info['bbox']
                        gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                        gt_labels.append(str(ann_info['category_id']))
                    item = {
                        'id': ann['id'],
                        'image': sub_dataset['img_path'] + ann['file_name'],
                        'dataset': sub_dataset_name,
                        'query': copy.deepcopy(sub_dataset['classes_en']),
                        'gt_labels': gt_labels,
                        'gt_bboxes': gt_bboxes,
                        'visual_prompt': visual_prompt,
                    }
                    self.ann.append(item)
        elif dataset == 'FG-OVD':
            self.proposals = json.load(open(ds_collections[dataset]['proposals']))
            for ann_path, sub_dataset, n_hardnegatives in zip(ds_collections[dataset]['ann_path'], ds_collections[dataset]['subset'], ds_collections[dataset]['n_hardnegatives']):
                data = json.load(open(ann_path))
                categories_done = []
                for ann in data['annotations']:
                    if ann['category_id'] not in categories_done:
                        categories_done.append(ann['category_id'])
                    else:
                        continue
                    # check if a number of hardnegatives is setted to non-default values
                    # if it is, the vocabulary is clipped and if it is too short, we skip that image
                    vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
                    if n_hardnegatives < 10:
                        len_vocabulary = n_hardnegatives + 1
                        if len(vocabulary) < len_vocabulary:
                            continue
                        vocabulary = vocabulary[:len_vocabulary]
                        vocabulary_id = vocabulary_id[:len_vocabulary]
                    image_filepath = get_image_filepath(ann['image_id'], data['images'])
                    item = {
                        'id': {
                            'category_id': ann['category_id'],
                            'vocabulary': vocabulary_id,
                            'image_filepath': image_filepath,
                        },
                        'image': ds_collections[dataset]['img_path'] + image_filepath,
                        'dataset': sub_dataset,
                        'query': vocabulary,
                        'gt_labels': vocabulary_id,
                        'gt_bboxes': [],
                        'visual_prompt': visual_prompt,
                    }
                    self.ann.append(item)
                

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ann = self.ann[idx]

        data = {}
        data['id'] = ann['id']
        data['image'] = Image.open(ann['image']).convert('RGB')
        w, h = data['image'].size
        if len(self.proposals[ann['image']]) == 2 and self.dataset != 'humanref':
            data['proposals'] = self.proposals[ann['image']][0][:100]
            data['propsoals_score'] = self.proposals[ann['image']][1][:100]
        else:
            data['proposals'] = self.proposals[ann['image']][:100]
        # data['proposals'] = ann['gt_bboxes'] + data['proposals']
        # data['proposals'] = data['proposals'][:100]
        # random.shuffle(data['proposals'])
        for i in range(len(data['proposals'])):
            data['proposals'][i][0] = max(0, min(w, data['proposals'][i][0]))
            data['proposals'][i][1] = max(0, min(h, data['proposals'][i][1]))
            data['proposals'][i][2] = max(0, min(w, data['proposals'][i][2]))
            data['proposals'][i][3] = max(0, min(h, data['proposals'][i][3]))

        if self.dataset == 'coco' or self.dataset == 'lvis' or self.dataset == 'coco_o':
            data['query'] = copy.deepcopy(self.query)
        elif self.dataset == 'refcoco' or self.dataset == 'odinw35' or self.dataset == 'odinw13' or self.dataset == 'd3' or self.dataset == 'FG-OVD':
            data['query'] = ann['query']
            
        data['dataset'] = ann['dataset']
        data['gt_labels'] = ann['gt_labels']
        data['gt_bboxes'] = ann['gt_bboxes']
        for i in range(len(data['gt_bboxes'])):
            data['gt_bboxes'][i][0] = max(0, min(w, data['gt_bboxes'][i][0]))
            data['gt_bboxes'][i][1] = max(0, min(h, data['gt_bboxes'][i][1]))
            data['gt_bboxes'][i][2] = max(0, min(w, data['gt_bboxes'][i][2]))
            data['gt_bboxes'][i][3] = max(0, min(h, data['gt_bboxes'][i][3]))
        data['visual_prompt'] = ann['visual_prompt']

        return data


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def collate_fn(inputs):
    return inputs


def eval_coco(ids, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    coco_gt = COCO(ds_collections['coco']['ann_path'])

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

    # 转换为COCO结果格式
    results = []
    for img_id, pred_bbox, pred_label, pred_score in zip(ids, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            results.append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    # 评估指标计算
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_coco_o(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    dataset2coco = {}
    for ann_path, sub_dataset in zip(ds_collections['coco_o']['ann_path'], ds_collections['coco_o']['subsets']):
        dataset2coco[sub_dataset] = COCO(ann_path)

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

    dataset2results = {sub_dataset_name: [] for sub_dataset_name in ds_collections['coco_o']['subsets']}
    # 转换为COCO结果格式
    for img_id, dataset, pred_bbox, pred_label, pred_score in zip(ids, datasets, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            dataset2results[dataset].append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    avg_map = []
    for sub_dataset_name, results in dataset2results.items():
        print(f"Evaluating {sub_dataset_name}...")
        coco_gt = dataset2coco[sub_dataset_name]
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        avg_map.append(coco_eval.stats[0])
    
    print(f"Average mAP across all sub-datasets: {np.mean(avg_map)}")


def _merge_lists(listA, listB, maxN, key):
    result = []
    indA, indB = 0, 0
    while (indA < len(listA) or indB < len(listB)) and len(result) < maxN:
        if (indB < len(listB)) and (indA >= len(listA)
                                    or key(listA[indA]) < key(listB[indB])):
            result.append(listB[indB])
            indB += 1
        else:
            result.append(listA[indA])
            indA += 1
    return result

def eval_lvis(ids, pred_bboxes, pred_labels, pred_scores):
    # 加载LVIS标注
    lvis_gt = LVIS(ds_collections['lvis']['ann_path'])

    # 转换为LVIS结果格式
    cur_results = []
    for img_id, pred_bbox, pred_label, pred_score in zip(ids, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            cur_results.append({
                "image_id": img_id,
                "category_id": label.item() + 1,
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })

    by_cat = defaultdict(list)
    for ann in cur_results:
        by_cat[ann['category_id']].append(ann)

    results = {}
    for cat, cat_anns in by_cat.items():
        if cat not in results:
            results[cat] = []

        cur = sorted(
            cat_anns, key=lambda x: x['score'], reverse=True)[:10000]
        results[cat] = _merge_lists(
            results[cat], cur, 10000, key=lambda x: x['score'])
    
    new_results = []
    missing_dets_cats = set()
    for cat, cat_anns in results.items():
        if len(cat_anns) < 10000:
            missing_dets_cats.add(cat)
        new_results.extend(
            sorted(cat_anns, key=lambda x: x['score'],
                    reverse=True)[:10000])

    if missing_dets_cats:
        print(
            f'\n===\n'
            f'{len(missing_dets_cats)} classes had less than {10000} '
            f'detections!\n Outputting {10000} detections for each '
            f'class will improve AP further.\n ===')

    new_results = LVISResults(lvis_gt, new_results, max_dets=-1)
    lvis_eval = LVISEval(lvis_gt, new_results, iou_type='bbox')
    params = lvis_eval.params
    params.max_dets = -1  # No limit on detections per image.
    lvis_eval.run()
    lvis_eval.print_results()



def eval_odinw35(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    dataset2coco = {}
    for sub_dataset_name, sub_dataset in ds_collections['odinw35']['datasets'].items():
        dataset2coco[sub_dataset_name] = COCO(sub_dataset['ann_path'])

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79}

    dataset2results = {sub_dataset_name: [] for sub_dataset_name in ds_collections['odinw35']['datasets'].keys()}

    # 转换为COCO结果格式
    for img_id, dataset, pred_bbox, pred_label, pred_score in zip(ids, datasets, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            dataset2results[dataset].append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    avg_map = []
    for sub_dataset_name, results in dataset2results.items():
        print(f"Evaluating {sub_dataset_name}...")
        coco_gt = dataset2coco[sub_dataset_name]
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        avg_map.append(coco_eval.stats[0])
    
    print(f"Average mAP across all sub-datasets: {np.mean(avg_map)}")



def eval_odinw13(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    # 加载COCO标注
    dataset2coco = {}
    for sub_dataset_name, sub_dataset in ds_collections['odinw13']['datasets'].items():
        dataset2coco[sub_dataset_name] = COCO(sub_dataset['ann_path'])

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 44, 44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 66, 66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79}

    dataset2results = {sub_dataset_name: [] for sub_dataset_name in ds_collections['odinw13']['datasets'].keys()}

    # 转换为COCO结果格式
    for img_id, dataset, pred_bbox, pred_label, pred_score in zip(ids, datasets, pred_bboxes, pred_labels, pred_scores):
        for box, label, score in zip(pred_bbox, pred_label, pred_score):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            w = xmax - xmin
            h = ymax - ymin
            
            dataset2results[dataset].append({
                "image_id": img_id,
                "category_id": id_map[label.item()],
                "bbox": [xmin, ymin, w, h],
                "score": score.item()
            })
    
    avg_map = []
    for sub_dataset_name, results in dataset2results.items():
        print(f"Evaluating {sub_dataset_name}...")
        coco_gt = dataset2coco[sub_dataset_name]
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        avg_map.append(coco_eval.stats[0])
    
    print(f"Average mAP across all sub-datasets: {np.mean(avg_map)}")



def eval_refcoco(ids, datasets, pred_bboxes, pred_labels, pred_scores):
    topk = (1, 5, 10)
    iou_thrs = 0.5
    dataset2score = {
        'refcoco_validation': {k: 0.0 for k in topk},
        'refcoco_test': {k: 0.0 for k in topk},
        'refcoco_testB': {k: 0.0 for k in topk},
        'refcocoplus_validation': {k: 0.0 for k in topk},
        'refcocoplus_test': {k: 0.0 for k in topk},
        'refcocoplus_testB': {k: 0.0 for k in topk},
        'refcocog_validation': {k: 0.0 for k in topk},
        'refcocog_test': {k: 0.0 for k in topk},
    }
    dataset2count = {
        'refcoco_validation': 0.0, 
        'refcoco_test': 0.0, 
        'refcoco_testB': 0.0, 
        'refcocoplus_validation': 0.0, 
        'refcocoplus_test': 0.0, 
        'refcocoplus_testB': 0.0, 
        'refcocog_validation': 0.0, 
        'refcocog_test': 0.0, 
    }

    # refcoco_validation
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcoco_validation':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)

    wrong_samples = []
    with open('datasets/refcoco/refcoco_validation.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcoco_validation'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        
        dataset2count['refcoco_validation'] += 1.0


    # refcoco_test
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcoco_test':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcoco/refcoco_test.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcoco_test'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcoco_test'] += 1.0

    
    # refcoco_testB
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcoco_testB':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcoco/refcoco_testB.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcoco_testB'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcoco_testB'] += 1.0


    # refcocoplus_validation
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocoplus_validation':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcocoplus/refcocoplus_validation.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocoplus_validation'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcocoplus_validation'] += 1.0


    # refcocoplus_test
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocoplus_test':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcocoplus/refcocoplus_test.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocoplus_test'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcocoplus_test'] += 1.0

    
    # refcocoplus_testB
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocoplus_testB':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcocoplus/refcocoplus_testB.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocoplus_testB'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcocoplus_testB'] += 1.0

    
    # refcocog_validation
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocog_validation':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcocog/refcocog_validation.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocog_validation'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcocog_validation'] += 1.0


    # refcocog_test
    subset_ids, subset_bboxes = [], []
    for img_id, dataset, pred_bbox in zip(ids, datasets, pred_bboxes):
        if dataset == 'refcocog_test':
            subset_ids.append(img_id)
            subset_bboxes.append(pred_bbox)
    
    with open('datasets/refcocog/refcocog_test.json') as f:
        data = json.load(f)
    
    gts = {}
    for da in data:
        gts[da['id']] = [da['bounding_boxes'], da["conversations"][1]["value"], da['image']]
    
    for img_id, pred_bbox in zip(subset_ids, subset_bboxes):
        target_bbox = gts[img_id][0]
        converted_bbox = pred_bbox.cpu().numpy()
        iou = bbox_overlaps(converted_bbox, np.array(target_bbox).reshape(-1, 4))
        for k in topk:
            if max(iou[:k]) >= iou_thrs:
                dataset2score['refcocog_test'][k] += 1.0
        if not max(iou[:1]) >= iou_thrs:
            wrong_samples.append({
                'image': gts[img_id][2],
                'label': gts[img_id][1],
                'pred_bbox': converted_bbox[:1].tolist(),
                'target_bbox': list(target_bbox)
            })
        dataset2count['refcocog_test'] += 1.0

    
    # summary
    for key, value in dataset2score.items():
        for k in topk:
            try:
                value[k] /= dataset2count[key]
            except Exception as e:
                print(e)

    for key, value in dataset2score.items():
        print(f' Dataset: {key} - Precision @ 1, 5, 10: {sorted([v for k, v in value.items()])}')




def eval_d3(ids, datasets, pred_bboxes, pred_labels, pred_scores):

    results_val = [{'img_id': img_id, "bboxes": pred_bbox.cpu().numpy(), "scores": pred_score.cpu(), 'labels': pred_label.cpu().numpy()} for img_id, pred_bbox, pred_score, pred_label, sub_dataset in zip(ids, pred_bboxes, pred_scores, pred_labels, datasets) if sub_dataset == 'FULL']

    results_testA = [{'img_id': img_id, "bboxes": pred_bbox.cpu(), "scores": pred_score.cpu(), 'labels': pred_label.cpu().numpy()} for img_id, pred_bbox, pred_score, pred_label, sub_dataset in zip(ids, pred_bboxes, pred_scores, pred_labels, datasets) if sub_dataset == 'PRES']

    results_testB = [{'img_id': img_id, "bboxes": pred_bbox.cpu(), "scores": pred_score.cpu(), 'labels': pred_label.cpu().numpy()} for img_id, pred_bbox, pred_score, pred_label, sub_dataset in zip(ids, pred_bboxes, pred_scores, pred_labels, datasets) if sub_dataset == 'ABS']
    
    from dod_metric import DODCocoMetric
    evaluator_val = DODCocoMetric(ds_collections['d3']['ann_path'][0])
    results = evaluator_val.compute_metrics(results_val)
    print("d3_FULL")
    print(results)

    evaluator_testA = DODCocoMetric(ds_collections['d3']['ann_path'][1])
    results = evaluator_testA.compute_metrics(results_testA)
    print("d3_PRES")
    print(results)

    evaluator_testB = DODCocoMetric(ds_collections['d3']['ann_path'][2])
    results = evaluator_testB.compute_metrics(results_testB)
    print("d3_ABS")
    print(results)


def convert_format(boxes):
    for box in boxes:
        box[2] += box[0]
        box[3] += box[1]
    return boxes

def get_image_ground_truth(data, image_id):
    """
    Given a dictionary 'data' and an 'image_id', returns a dictionary with 'boxes' and 'categories' information for
    that image.

    Args:
        data (dict): The data dictionary containing 'annotations'.
        image_id (int): The image_id for which to retrieve data.

    Returns:
        dict: A dictionary with 'boxes' and 'categories' information for the given image_id.
    """
    image_data = {'boxes': [], 'labels': []}  # Initialize the dictionary to store image data

    # Loop through each annotation in the 'annotations' list
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # If the 'image_id' in the annotation matches the given 'image_id', append bbox and category_id to the lists
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    # tensorize elements
    image_data['boxes'] = torch.tensor(image_data['boxes']).float()
    image_data['labels'] = torch.tensor(image_data['labels']).int()
    
    return image_data


def get_image_preds(preds):
    labels = []
    scores = []
    boxes = []
    for pred in preds:
        labels += [x for x in pred['labels']]
        scores += [x for x in pred['scores']]
        boxes += ([x for x in pred['boxes']])

    boxes = boxes if boxes != [] else [[0,0,0,0]]
    return {
        'boxes': torch.tensor(boxes).float(),
        'labels': torch.tensor(labels).int(),
        'scores': torch.tensor(scores).float()
    }


def apply_NMS(preds, iou=0.5):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    
    indexes_to_keep = batched_nms(boxes, 
                                  scores, 
                                  torch.tensor([0] * len(boxes)),
                                  iou)

    preds['boxes'] = boxes[indexes_to_keep]
    preds['scores'] = scores[indexes_to_keep]
    preds['labels'] = labels[indexes_to_keep]
    return preds

def eval_fg_ovd(ids, datasets, pred_bboxes, pred_labels, pred_scores):

    pred_result = {'Hard': {}, 'Medium': {}, 'Easy': {}, 'Trivial': {}, 'Color': {}, 'Material': {}, 'Pattern': {}, 'Transparancy': {}}
    for img_id, sub_dataset, pred_bbox, pred_label, pred_score in zip(ids, datasets, pred_bboxes, pred_labels, pred_scores):
        image = img_id['image_filepath']
        if image not in pred_result[sub_dataset]:
            pred_result[sub_dataset][image] = []
        img_id['boxes'] = pred_bbox.cpu().tolist()
        img_id['labels'] = pred_label.cpu().tolist()
        img_id['scores'] = pred_score.cpu().tolist()
        pred_result[sub_dataset][image].append(img_id)
    
    for sub_dataset, ann_path, n_neg in zip(ds_collections['FG-OVD']['subset'], ds_collections['FG-OVD']['ann_path'], ds_collections['FG-OVD']['n_hardnegatives']):
        with open(ann_path) as f:
            test_set = json.load(f)
        # Initialize metric
        from torchmetrics.detection import MeanAveragePrecision
        metric = MeanAveragePrecision(sync_on_compute=False)

        targets = []
        preds = []
        
        n_images = 0
        # for imm in tqdm(test_set['images']):
        for imm in test_set['images']:
            target = get_image_ground_truth(test_set, imm['id'])
 
            if imm['file_name'] in pred_result[sub_dataset]:
                # if args.disable_nms:
                #     pred = get_image_preds(preds_per_image[imm['file_name']])
                # else:
                # in case the ground truth for the image includes captions not processed by the detector, we remove them
                relevant_cats = [predictions['category_id'] for predictions in pred_result[sub_dataset][imm['file_name']]]
                mask = torch.isin(target['labels'], torch.tensor(relevant_cats))
                target['labels'] = target['labels'][mask]
                target['boxes'] = target['boxes'][mask]
                preds_per_cat = [get_image_preds([pred_per_cat]) for pred_per_cat in pred_result[sub_dataset][imm['file_name']]]
                preds_per_cat = [apply_NMS(pred_per_cat) for pred_per_cat in preds_per_cat]
                pred = {
                    'boxes': torch.cat([x['boxes'] for x in preds_per_cat], dim=0),
                    'labels': torch.cat([x['labels'] for x in preds_per_cat], dim=0),
                    'scores': torch.cat([x['scores'] for x in preds_per_cat], dim=0),
                }
            else:
                continue
            n_images += 1
            targets.append(target)
            preds.append(pred)
            
        # Update metric with predictions and respective ground truth
        metric.update(preds, targets)
        
        # getting time of execution of the mAP
        print("Starting mAP computation")
        # Compute the results
        result = metric.compute()
        # print("--- %s seconds ---" % (time.time() - start_time))
        result['n_images'] = n_images
        # de-tensorize the results:
        result = {
            'map': float(result['map']),
            'map_50': float(result['map_50']),
            'map_75': float(result['map_75']),
            'map_small': float(result['map_small']),
            'map_medium': float(result['map_medium']),
            'map_large': float(result['map_large']),
            'mar_1': float(result['mar_1']),
            'mar_10': float(result['mar_10']),
            'mar_100': float(result['mar_100']),
            'mar_small': float(result['mar_small']),
            'mar_medium': float(result['mar_medium']),
            'mar_large': float(result['mar_large']),
            'map_per_class': float(result['map_per_class']),
            'mar_100_per_class': float(result['mar_100_per_class']),
            'n_images': int(result['n_images'])  
        }
        
        # print(result)
        print(f"Done {sub_dataset}. {result}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_select', type=int, default=300)
    parser.add_argument('--nms', action='store_true')
    parser.add_argument('--score_thre', type=float, default=-1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--single_box', action='store_true')
    parser.add_argument('--task_specific_visual_prompt', action='store_true')
    args = parser.parse_args()


    from datetime import timedelta
    timeout = timedelta(seconds=7200)
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=timeout,
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    from models.vision_process import process_vision_info
    from transformers import AutoProcessor

    # Model initialization
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='cuda',
    )

    from models.qwen3vl_objembed import ObjectEmbed
    model = ObjectEmbed.from_pretrained(args.checkpoint, **model_kwargs)
    model = model.eval()

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    local_text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")
    model.model.object_token_id = object_token_index
    global_id = None
    global_text_id = None
    if model.use_global_caption:
        global_id = processor.tokenizer.convert_tokens_to_ids("<global>")
        global_text_id = processor.tokenizer.convert_tokens_to_ids("<global_text>")

    random.seed(args.seed)
    dataset = GroundingDataset(args.dataset, args.task_specific_visual_prompt)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    image_ids = []
    image_datasets = []
    all_pred_bboxes = []
    all_pred_labels = []
    all_pred_scores = []
    visualize_idx = 0
    text_embeddings = None
    for inputs in tqdm(dataloader, disable=torch.distributed.get_rank() != 0):
        image_ids.append(inputs[0]['id'])
        image_datasets.append(inputs[0]['dataset'])
        image = inputs[0]['image']
        ori_shape = [image.size]
        proposals = copy.deepcopy(inputs[0]['proposals'])
        proposals = [torch.tensor(proposals).cuda().to(model.dtype)]
        if 'propsoals_score' in inputs[0]:
            propsoals_score = torch.tensor(inputs[0]['propsoals_score'])

        if (args.dataset == 'coco' or args.dataset == 'lvis' or args.dataset == 'coco_o') and text_embeddings is not None:
            pass
        else:
            text_embeddings = []
            for i, prompt in enumerate(inputs[0]['query']):
                messages = [
                    {
                        "role": "user", 
                        "content": 
                        [
                            {"type": "text", "text": "Find an object that matches the given caption. %s <local_text>" % (prompt)}
                        ]
                    }
                ]
                texts = [processor.apply_chat_template(messages, tokenize=False).strip()]
                model_inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    do_resize=False,
                )
                model_inputs = model_inputs.to(model.device)
                
                with torch.inference_mode():
                    pred = model(
                        text_processed=model_inputs,
                        global_id=global_id,
                        local_text_id=local_text_id,
                        global_text_id=global_text_id
                    )
                text_embeddings.append(pred['local_text_embeddings'])
                
            text_embeddings = torch.cat(text_embeddings)
            text_embeddings = F.normalize(text_embeddings)

        obj_str = ""
        for j in range(proposals[0].shape[0]):
            obj_str += "Object %d: <object><object>. " % j
            
        if model.use_two_tokens == 0:
            obj_str = obj_str + "The global image is <global>"
        elif model.use_two_tokens == 1:
            obj_str = "The global image is <global>. " + obj_str + "The global image is <global>"
        else:
            obj_str = "The coarse global image is <global>. " + obj_str + " The detailed global image is <global>. "
        messages = [
            {
                "role": "user", 
                "content": 
                [
                    {"type": "image", "image": image}, 
                    {"type": "text", "text": inputs[0]['visual_prompt'] + obj_str}
                ]
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        texts = [processor.apply_chat_template(messages, tokenize=False).strip()]
        model_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        model_inputs = model_inputs.to(model.device)
        
        with torch.inference_mode():
            pred = model(
                **model_inputs,
                bboxes=copy.deepcopy(proposals),
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                global_id=global_id,
                local_text_id=local_text_id,
                global_text_id=global_text_id
            )

        object_embeddings = pred['object_embeddings']
        object_embeddings = F.normalize(object_embeddings)

        pred_scores = object_embeddings @ text_embeddings.transpose(-1, -2)
        pred_scores = pred_scores * model.logit_log_scale.exp()
        pred_scores = pred_scores + model.logit_bias
        pred_scores = pred_scores.float().sigmoid().cpu()
        objectness = pred['objness'].sigmoid().repeat(1, len(text_embeddings)).cpu().float()
        pred_scores *= objectness
        pred_labels = torch.arange(len(text_embeddings)).unsqueeze(0).repeat(len(object_embeddings), 1).cpu()
        pred_bboxes = proposals[0].unsqueeze(1).repeat(1, len(text_embeddings), 1).cpu().float()

        if not args.single_box:
            pred_bboxes = pred_bboxes.flatten(0, 1)
            pred_labels = pred_labels.flatten()
            pred_scores = pred_scores.flatten(0, 1)
            if len(pred_bboxes) > 1000:
                topk_values, topk_indexes = torch.topk(
                    pred_scores.view(-1), 1000, dim=0)
                pred_scores = topk_values
                pred_bboxes = pred_bboxes[topk_indexes]
                pred_labels = pred_labels[topk_indexes]
            if args.nms:
                selected_indices = torchvision.ops.batched_nms(pred_bboxes, pred_scores, pred_labels, iou_threshold=0.7)
                pred_bboxes = pred_bboxes[selected_indices]
                pred_labels = pred_labels[selected_indices]
                pred_scores = pred_scores[selected_indices]
            if args.score_thre > 0:
                mask = pred_scores > args.score_thre
                pred_bboxes = pred_bboxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]
            else:
                topk = min(args.num_select, len(pred_scores))
                topk_values, topk_indexes = torch.topk(
                    pred_scores.view(-1), topk, dim=0)
                pred_scores = topk_values
                pred_bboxes = pred_bboxes[topk_indexes]
                pred_labels = pred_labels[topk_indexes]
        
        else:
            pred_bboxes = pred_bboxes[:, 0, :]
            pred_scores, pred_labels = torch.max(pred_scores, dim=1)
            if args.nms:
                selected_indices = torchvision.ops.batched_nms(pred_bboxes, pred_scores, pred_labels, iou_threshold=0.7)
                pred_bboxes = pred_bboxes[selected_indices]
                pred_labels = pred_labels[selected_indices]
                pred_scores = pred_scores[selected_indices]

        if args.dataset == 'd3' or args.dataset == 'FG-OVD':
            gt_labeles = torch.tensor(inputs[0]['gt_labels'])
            pred_labels = gt_labeles[pred_labels.cpu()]
            
        all_pred_bboxes.append(pred_bboxes)
        all_pred_labels.append(pred_labels)
        all_pred_scores.append(pred_scores)
 
            

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_ids = [None for _ in range(world_size)]
    merged_image_datasets = [None for _ in range(world_size)]
    merged_pred_bboxes = [None for _ in range(world_size)]
    merged_pred_labels = [None for _ in range(world_size)]
    merged_pred_scores = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_ids, image_ids)
    torch.distributed.all_gather_object(merged_image_datasets, image_datasets)
    torch.distributed.all_gather_object(merged_pred_bboxes, all_pred_bboxes)
    torch.distributed.all_gather_object(merged_pred_labels, all_pred_labels)
    torch.distributed.all_gather_object(merged_pred_scores, all_pred_scores)

    merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
    merged_image_datasets = [_ for _ in itertools.chain.from_iterable(merged_image_datasets)]
    merged_pred_bboxes = [_ for _ in itertools.chain.from_iterable(merged_pred_bboxes)]
    merged_pred_labels = [_ for _ in itertools.chain.from_iterable(merged_pred_labels)]
    merged_pred_scores = [_ for _ in itertools.chain.from_iterable(merged_pred_scores)]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.dataset} ...")
        if args.dataset == 'coco':
            eval_coco(merged_ids, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'lvis':
            eval_lvis(merged_ids, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'coco_o':
            eval_coco_o(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'refcoco':
            eval_refcoco(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'odinw35':
            eval_odinw35(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'odinw13':
            eval_odinw13(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'd3':
            eval_d3(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        elif args.dataset == 'FG-OVD':
            eval_fg_ovd(merged_ids, merged_image_datasets, merged_pred_bboxes, merged_pred_labels, merged_pred_scores)
        
        
    torch.distributed.barrier()


