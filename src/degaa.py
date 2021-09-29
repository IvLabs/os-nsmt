import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset

sys.path.append(os.getcwd())
from dalib.modules.domain_discriminator import DomainDiscriminator
from common.modules.classifier import Classifier
#from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
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

from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Total GPUs Used:", torch.cuda.device_count())
i = 0
print("Hardwares Used: ")
while(i < torch.cuda.device_count()):
  print(torch.cuda.get_device_name(i))
  i = i + 1

def main(args):
  #logger = CompleteLogger(args.log, args.phase)
  normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

  train_transform = T.Compose([ResizeImage(256),
  T.CenterCrop(224),
  T.RandomHorizontalFlip(),
  T.ToTensor(),
  normalize
  ])

  val_transform = T.Compose([ResizeImage(256),
  T.CenterCrop(224),
  T.ToTensor(),
  normalize
  ])

  dataset = datasets.__dict__[args.dataset]
  source_dataset = open_set(dataset, source = True)
  target_dataset = open_set(dataset, source = False)
  # OfficeHome Ar: Art, Cl: Clipart, Pr: Product, Rw: Real - World
  #print(args.source, args.target, dataset.domains())
  #print(args.source.split(','))
  args.source = args.source.split(',')
  args.target = args.target.split(',')
  if args.dataset == 'OfficeHome':
    args.root = os.path.join(args.root, 'office-home')
  train_source_dataset = ConcatDataset([source_dataset(root=args.root, task=source, download=True, transform=train_transform) 
  for source in args.source])
  train_target_dataset = ConcatDataset([target_dataset(root=args.root, task=target, download=True, transform=train_transform) 
  for target in args.target])

  train_source_loader = DataLoader(train_source_dataset, batch_size = args.batch_size, shuffle = True,
  num_workers = args.workers, drop_last = True)
  train_target_loader = DataLoader(train_target_dataset, batch_size = args.batch_size, shuffle = True,
  num_workers = args.workers, drop_last = True)

  val_dataset = dataset(root=args.root, task = args.target[0], download = True, transform = val_transform)
  val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)
  
  if args.dataset == 'DomainNet':
    test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  else:
    test_loader = val_loader 

  train_source_iter = ForeverDataIterator(train_source_loader)
  train_target_iter = ForeverDataIterator(train_target_loader)

  backbone = models.__dict__[args.arch](pretrained = True).to(device)
  num_classes = val_dataset.num_classes

  classifier = ImageClassifier(backbone, num_classes = args.bottleneck_dim, bottleneck_dim = args.bottleneck_dim).to(device)
  #summary(classifier, (3, 224, 224))

  optimizer = SGD(classifier.get_parameters(), args.lr, momentum = args.momentum, weight_decay = args.wd, nesterov = True)
  lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1 + args.lr_gamma * float(x)) ** (-args.lr_decay))

  gaa = GAA(input_dim = args.bottleneck_dim, gnn_layers = 6, num_heads = 4).to(device)
  #summary(gaa, [(32, 1024),(32, 1024)])

  # Running for one epoch as of now
  f_s, f_t = train(train_source_iter, train_target_iter, classifier, gaa, optimizer, lr_scheduler, 0, args) # Add epoch instead of 0 in loss
  print(f_s.shape, f_t.shape)

  #if args.phase == 'test':
  #  checkpo

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier, gaa: GAA, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
  model.train()
  gaa.train()
  x_s, label_s = next(train_source_iter)
  x_t, label_t = next(train_target_iter)
  
  x_s = x_s.to(device)
  label_s = label_s.to(device)
  x_t = x_t.to(device)
  label_t = label_t.to(device)
  x = torch.cat((x_s, x_t), dim=0)
  y, f = model(x)
  y_s, y_t = y.chunk(2, dim=0)
  f_s, f_t = f.chunk(2, dim=0)

  f_s, f_t = gaa(f_s, f_t)

  return f_s, f_t



if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description = "DEGAA")

  parser.add_argument('-d', '--dataset', default = 'OfficeHome')
  parser.add_argument('-a', '--arch', default = 'resnet50')
  parser.add_argument('-b', '--batch_size', default = 32)
  parser.add_argument('--lr', '--learning_rate', default = 0.002)
  parser.add_argument('--lr-gamma', default = 0.001)
  parser.add_argument('--bottleneck-dim', default = 1024)
  parser.add_argument('--lr-decay', default = 0.75)
  parser.add_argument('--momentum', default = 0.9)
  parser.add_argument('--wd', '--weight-decay', default = 1e-3)
  parser.add_argument('-j', '--workers', default = 2)
  parser.add_argument('--epochs', default = 20)
  parser.add_argument('--root', default = 'data')
  parser.add_argument('-s', '--source', help = 'source domain(s)')
  parser.add_argument('-t', '--target', help = 'target domain(s)')

  args = parser.parse_args()
  main(args)
