import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import numpy as np

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

#from torchsummary import summary

import wandb

wandb.init(project = "degaa")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Total GPUs Used:", torch.cuda.device_count())
i = 0
print("Hardwares Used: ")
while(i < torch.cuda.device_count()):
  print(torch.cuda.get_device_name(i))
  i = i + 1

def main(args: argparse.Namespace):
  
  wandb.run.name = 'run_' + str(args.source) + str(args.target)

  logger = CompleteLogger(args.log, args.phase)

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    cudnn.benchmark = True

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
  #args.source = args.source.split(',')
  #args.target = args.target.split(',')
  if args.dataset == 'OfficeHome':
    args.root = os.path.join(args.root, 'office-home')
  #train_source_dataset = ConcatDataset([source_dataset(root=args.root, task=source, download=True, transform=train_transform) for source in args.source])
  #train_target_dataset = ConcatDataset([target_dataset(root=args.root, task=target, download=True, transform=train_transform) for target in args.target])
  train_source_dataset = source_dataset(root=args.root, task=args.source, download=True, transform=train_transform)
  train_target_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=train_transform)

  train_source_loader = DataLoader(train_source_dataset, batch_size = args.batch_size, shuffle = True,
  num_workers = args.workers, drop_last = True)
  train_target_loader = DataLoader(train_target_dataset, batch_size = args.batch_size, shuffle = True,
  num_workers = args.workers, drop_last = True)

  val_dataset = target_dataset(root=args.root, task = args.target, download = True, transform = val_transform)
  val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.workers)
  
  if args.dataset == 'DomainNet':
    test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  else:
    test_loader = val_loader 

  train_source_iter = ForeverDataIterator(train_source_loader)
  train_target_iter = ForeverDataIterator(train_target_loader)

  #val_iter = ForeverDataIterator(val_loader)

  backbone = models.__dict__[args.arch](pretrained = True).to(device)
  num_classes = val_dataset.num_classes

  classifier = ImageClassifier(backbone, num_classes = num_classes, bottleneck_dim = args.bottleneck_dim).to(device)
  #summary(classifier, (3, 224, 224))

  optimizer = SGD(classifier.get_parameters(), args.lr, momentum = args.momentum, weight_decay = args.wd, nesterov = True)
  lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1 + args.lr_gamma * float(x)) ** (-args.lr_decay))

  gaa = GAA(input_dim = args.bottleneck_dim, num_classes = num_classes, gnn_layers = 6, num_heads = 4).to(device)
  #summary(gaa, [(32, 1024),(32, 1024)])

  if args.phase == 'test':
    acc1 = validate(test_loader, classifier, args)
    print(acc1)
    return

  best_h_score = 0.
  for epoch in range(args.epochs):
    train(train_source_iter, train_target_iter, classifier, gaa, optimizer, lr_scheduler, epoch, args)
    h_score = validate(val_loader, classifier, args)

    torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
    if h_score > best_h_score:
      shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
  
  print("best_h_score = {:3.1f}".format(best_h_score))

  classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
  h_score = validate(test_loader, classifier, args)
  print("test_h_score = {:3.1f}".format(h_score))

  logger.close()

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier, gaa: GAA, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, num_classes = 65):
  
  batch_time = AverageMeter('Time', ':5.2f')
  data_time = AverageMeter('Data', ':5.2f')
  losses = AverageMeter('Loss', ':6.2f')
  cls_accs = AverageMeter('Cls Acc', ':3.1f')
  tgt_accs = AverageMeter('Tgt Acc', ':3.1f')
  progress = ProgressMeter(args.iters_per_epoch, [batch_time, data_time, losses, cls_accs, tgt_accs],
  prefix = "Epoch: [{}]".format(epoch))
  
  
  model.train()
  gaa.train()
  clf = LocalOutlierFactor(n_neighbors=num_classes + 1, contamination=0.1)
  end = time.time()
  for i in range(args.iters_per_epoch):
    x_s, label_s = next(train_source_iter)
    x_t, label_t = next(train_target_iter)
    
    x_s = x_s.to(device)
    label_s = label_s.to(device)
    x_t = x_t.to(device)
    label_t = label_t.to(device)

    data_time.update(time.time() - end)

    x = torch.cat((x_s, x_t), dim=0)
    y, f = model(x)
    y_s, y_t = y.chunk(2, dim=0)
    f_s, f_t = f.chunk(2, dim=0)

    f_t_numpy = f_t.clone().cpu().detach().numpy()
    y_pred = clf.fit_predict(f_t_numpy)
    index = np.where(y_pred==-1)
    label_t[index] = num_classes + 1

    """
    get centroids and find classes.
    """


    f_s, f_t, y_s, y_t = gaa(f_s, f_t)
    y_s, y_t = nn.Softmax(dim = 1)(y_s), nn.Softmax(dim = 1)(y_t)

    cls_loss_s, cls_loss_t = F.cross_entropy(y_s, label_s), F.cross_entropy(y_t, label_t)
    loss = cls_loss_s + cls_loss_t * args.trade_off

    cls_acc = accuracy(y_s, label_s)[0]
    tgt_acc = accuracy(y_t, label_t)[0]

    losses.update(loss.item(), x_s.size(0))
    cls_accs.update(cls_acc.item(), x_s.size(0))
    tgt_accs.update(tgt_acc.item(), x_s.size(0))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    batch_time.update(time.time() - end)
    end = time.time()

    wandb.log({'accuracy_source': cls_acc, 'accuracy_target': tgt_acc, 'loss': loss})

    if i % args.print_freq == 0:
      progress.display(i)

def validate(val_loader: DataLoader, model: Classifier, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    classes = val_loader.dataset.classes
    confmat = ConfusionMatrix(len(classes))
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            softmax_output = F.softmax(output, dim=1)
            softmax_output[:, -1] = args.threshold

            # measure accuracy and record loss
            confmat.update(target, softmax_output.argmax(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        acc_global, accs, iu = confmat.compute()
        all_acc = torch.mean(accs).item() * 100
        known = torch.mean(accs[:-1]).item() * 100
        unknown = accs[-1].item() * 100
        h_score = 2 * known * unknown / (known + unknown)
        if args.per_class_eval:
            print(confmat.format(classes))
        print(' * All {all:.3f} Known {known:.3f} Unknown {unknown:.3f} H-score {h_score:.3f}'
              .format(all=all_acc, known=known, unknown=unknown, h_score=h_score))

    return h_score
    



if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description = "DEGAA")

  parser.add_argument('-d', '--dataset', default = 'OfficeHome')
  parser.add_argument('-a', '--arch', default = 'resnet50')
  parser.add_argument('-b', '--batch_size', type = int, default = 32)
  parser.add_argument('--lr', '--learning_rate', default = 0.002)
  parser.add_argument('--lr-gamma', default = 0.001)
  parser.add_argument('--bottleneck-dim', default = 2048)
  parser.add_argument('--lr-decay', default = 0.75)
  parser.add_argument('--momentum', default = 0.9)
  parser.add_argument('--wd', '--weight-decay', default = 1e-3)
  parser.add_argument('-j', '--workers', default = 2)
  parser.add_argument('--epochs', type = int, default = 20)
  parser.add_argument('--root', default = 'data')
  parser.add_argument('-s', '--source', help = 'source domain(s)')
  parser.add_argument('-t', '--target', help = 'target domain(s)')
  parser.add_argument('--trade-off', default = 1., type = float)
  parser.add_argument('-i', '--iters-per-epoch', default = 500, type = int)
  parser.add_argument('-p', '--print-freq', default = 100, type = int)
  parser.add_argument('--log', type = str, default = 'degaa')
  parser.add_argument('--seed', default = None, type = int)
  parser.add_argument('--phase', type = str, default = 'train')
  parser.add_argument('--threshold', default = 0.8, type = float)
  parser.add_argument('--per-class-eval', action = 'store_true')

  args = parser.parse_args()
  main(args)

