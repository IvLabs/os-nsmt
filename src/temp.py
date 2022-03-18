import random
import time
from typing import Counter
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
sys.path.append("")
from common.modules.classifier import Classifier

# from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.adaptation.degaa import ImageClassifier, GAA
import common.vision.datasets.openset as datasets
from common.vision.datasets.openset import default_open_set as open_set
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
import network
from data_helper import setup_datasets
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = argparse.Namespace(dataset='OfficeHome', arch='resnet50', batch_size=32, lr=0.002, lr_gamma=0.001, bottleneck_dim=256, feature_dim=256, lr_decay=0.75, momentum=0.9, wd=0.001, workers=0, epochs=1000, root='./data', source='Ar,Pr', target='Cl,Rw', proto_path='./protoruns/run7/prototypes.pth', centroid_path='./centroids/OfficeHome/ArPr_centroid.npy', proto_dim=512, bottleneck=256, layer='wn', classifier='bn', trade_off=1.0, iters_per_epoch=500, print_freq=100, log='degaa', seed=0, phase='train', threshold=0.8, per_class_eval=False, tensorboard=False, wandb=False, output_dir='./adapt/run1', trained_wt='weights/oda', net='resnet50')


num_classes, train_source_loader, train_target_loader, val_loader, test_loader = setup_datasets(args, concat=True)

sample = next(iter(train_source_loader))
print("Sample.shape", sample[0][0].shape)

grid  = torchvision.utils.make_grid(sample[0][0], normalize=True)

plt.figure(figsize=(15,10))
plt.imshow(grid.permute(1,2,00))
plt.savefig("./src/tmep.png")
plt.show()


CLASSES = train_source_loader.dataset.datasets[0].CLASSES
'''
CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
'''
print(sample[0][1])
print([CLASSES[c] for c in sample[0][1]])

from common.vision.datasets.officehome import OfficeHome
from common.vision.datasets.Concatenate import ConcatenateDataset

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

source_datasets = [
    OfficeHome(root=args.root, task=source, download=True, transform=train_transform)
    for source in args.source
]
office = ConcatDataset(source_datasets)
loader = DataLoader(office, batch_size=32, shuffle=True)

sample = next(iter(loader))
print("Sample: ", sample[0].shape, "Sample Labels:", sample[1])
print([CLASSES[c] for c in sample[1]])

grid  = torchvision.utils.make_grid(sample[0], normalize=True)

plt.figure(figsize=(15,10))
plt.imshow(grid.permute(1,2,00))
plt.savefig("./src/tmp.png")
plt.show()

print("END")