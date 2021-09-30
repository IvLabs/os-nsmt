import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../../..')
from common.modules.classifier import Classifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

from sklearn.cluster import KMeans
import torch
import numpy as np
from data_list import ImageList, ImageList_idx
import network
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(style="darkgrid")

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_load(args, batch_size=64):
    ## prepare data
    txt_path=f'data/{args.dset}'

    
    def image(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    def image_test(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            normalize
        ])

    dsets = {}
    dset_loaders = {}
    dset_loaders_test = {}

    train_bs = batch_size
    
    if args.dset == 'office':
        txt_files = {'amazon' : f'{txt_path}/amazon.txt', 
                    'webcam': f'{txt_path}/webcam.txt', 
                    'dslr': f'{txt_path}/dslr.txt'}
        args.num_classes = 31

    if args.dset == 'office-home':
        txt_files = {'Art' : f'{txt_path}/Art.txt', 
                'Clipart': f'{txt_path}/Clipart.txt', 
                'Product': f'{txt_path}/Product.txt',
                'RealWorld': f'{txt_path}/RealWorld.txt'}
        args.num_classes = 65

    if args.dset == 'pacs':
        txt_files = {'art_painting' : f'{txt_path}/art_painting.txt', 
                'cartoon': f'{txt_path}/cartoon.txt', 
                'photo': f'{txt_path}/photo.txt',
                'sketch': f'{txt_path}/sketch.txt'}
    
    if args.dset == 'domain_net':
        txt_files = {'clipart': f'{txt_path}/clipart.txt',
                'infograph': f'{txt_path}/infograph.txt', 
                'painting':  f'{txt_path}/painting.txt', 
                'quickdraw': f'{txt_path}/quickdraw.txt', 
                'sketch':    f'{txt_path}/sketch.txt', 
                'real':      f'{txt_path}/real.txt'}

        txt_files_test = {'clipart': f'{txt_path}/clipart_test.txt',
                'infograph': f'{txt_path}/infograph_test.txt', 
                'painting':  f'{txt_path}/painting_test.txt', 
                'quickdraw': f'{txt_path}/quickdraw_test.txt', 
                'sketch':    f'{txt_path}/sketch_test.txt', 
                'real':      f'{txt_path}/real_test.txt'}
        args.num_classes = 345

        for domain, paths in txt_files_test.items(): 
            txt_tar = open(paths).readlines()
            dsets[domain] = ImageList_idx(txt_tar, transform=image_test())
            dset_loaders_test[domain] = DataLoader(dsets[domain], batch_size=train_bs,drop_last=False)

    if args.dset != 'domain_net':
        dset_loaders_test = dset_loaders
    
    for domain, paths in txt_files.items(): 
        if domain in [args.source, args.target]:
            txt_tar = open(paths).readlines()

            dsets[domain] = ImageList_idx(txt_tar, transform=image())
            dset_loaders[domain] = DataLoader(dsets[domain], batch_size=train_bs, shuffle=True,drop_last=True)

    return dset_loaders, dset_loaders_test


def load_models(args,domain): 

    model_loaders = {}
    if args.dset == 'office-home':
        wt_abbr = { 'Art': 'A','Clipart': 'C', 'Product': 'P', 'RealWorld': 'R'}

    dom_adapts = wt_abbr[domain]

    print('Loading weights for ', domain)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).to(device)
    elif args.net == 'vit':
        netF = network.ViT().to(device)
    
    args.feature_dim = netF.in_features
    args.bottleneck_dim = 256

    netB = network.feat_bootleneck(type='bn', feature_dim=args.feature_dim,bottleneck_dim=args.bottleneck_dim).to(device)
    netC = network.feat_classifier(type='wn', class_num=args.num_classes, bottleneck_dim=args.bottleneck_dim).to(device)

    modelpathF = f'{args.trained_wt}/{dom_adapts}/source_F.pt'
    netF.load_state_dict(torch.load(modelpathF))
    
    modelpathB = f'{args.trained_wt}/{dom_adapts}/source_B.pt'
    netB.load_state_dict(torch.load(modelpathB))

    modelpathC = f'{args.trained_wt}/{dom_adapts}/source_C.pt'
    netC.load_state_dict(torch.load(modelpathC))
    
    netF.eval()
    netB.eval()
    netC.eval()

    model_loaders =  [netF,netB, netC]
        # break
    
    print('Feature Extractors made Successfully !')
    return model_loaders


def compute_features(args, net, dataloader, dataset_name=None):
    # print(len(dataloader))
    netF,netB, netC = net[0], net[1], net[2]
    print(f'Computing features for [{dataset_name}] -', len(dataloader)*args.batch_size, 'images')

    correct = 0
    total = 0
    stored_feat_lbl = {}
    stored_feat_lbl['feature'] = torch.empty((1,args.bottleneck_dim)).to(device)
    stored_feat_lbl['label'] = torch.empty((1), dtype=torch.int).to(device)
    # print(store_feat)
    
    with torch.no_grad():
        iter_test = iter(dataloader)
        for i in tqdm(range(len(dataloader))):
            data = iter_test.next()
            inputs = data[0].to('cuda')
            labels = data[1].to('cuda')
            
            feats = netB(netF(inputs))            

            stored_feat_lbl['feature'] = torch.cat((stored_feat_lbl['feature'], feats), 0)
            stored_feat_lbl['label'] = torch.cat((stored_feat_lbl['label'], labels), 0)

            outputs = netC(feats)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if i == 1:
            #     break          
            
    accuracy = 100 * correct / total
    
    log_str ='Accuracy of the src net on the {} images: {}'.format(dataset_name, accuracy)

    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    print(log_str + '\n')

    stored_feat_lbl['feature'] = stored_feat_lbl['feature'][1:]
    stored_feat_lbl['label'] = stored_feat_lbl['label'][1:]

    return stored_feat_lbl

def compute_centroids(args,features_labels):
    
    run_sum = torch.zeros((args.num_classes, args.bottleneck_dim), dtype=torch.float).to(device)
    lbl_cnt = torch.ones((args.num_classes), dtype=torch.float).to(device)
    cls_centroids = torch.zeros((args.num_classes, args.bottleneck_dim), dtype=torch.float).to(device)

    for feat, lbl in zip(features_labels['feature'],features_labels['label']):
        run_sum[lbl] += feat
        lbl_cnt[lbl] += 1
        
    for i in range(args.num_classes):
        # if lbl_cnt[i] == 0:
        #     print('Failed computing centroid (no images present) for Class', i)
        cls_centroids[i] = run_sum[i] / lbl_cnt[i]
    return cls_centroids
    # print(run_sum.shape)
    
def tsne_plotter(args, np_cls_centroids, np_feat, np_label, plt_name=None):

        cat_feat = np.concatenate((np_cls_centroids, np_feat), axis=0)
        data_cen, data_feat = {}, {}
        X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(cat_feat)
        data_cen['x_cent'], data_cen['y_cent'], data_cen['lbl_cent'] = X_embedded[:args.num_classes,0],X_embedded[:args.num_classes,1], np.arange(args.num_classes)
        data_feat['x_feat'], data_feat['y_feat'], data_feat['lbl_feat'] = X_embedded[args.num_classes:,0],X_embedded[args.num_classes:,1], np_label

        data_feat=pd.DataFrame(data_feat)
        data_cen=pd.DataFrame(data_cen)

        sns.scatterplot(data=data_feat, x="x_feat", y="y_feat", legend=False, hue='lbl_feat', 
                            palette=sns.color_palette("Spectral", as_cmap=True) , alpha=0.5, s=12)
        sns.scatterplot(data=data_cen, x="x_cent", y="y_cent", hue='lbl_cent', legend=False, 
                            marker='X', palette=sns.color_palette("Spectral", as_cmap=True))
        plt.savefig(f'tsne_plts/tsne_{plt_name}.pdf')

def main(args):

    dom_dataloaders, dset_loaders_test = data_load(args, batch_size=args.batch_size) 
    model_loaders ={}

    for domain in [args.source, args.target]:
        model_loaders[domain] = load_models(args, domain)
        features_labels = {}

        # Compute features and labels of src and targets using src trained wts
        features_labels[domain] = compute_features(args, model_loaders[domain] , dom_dataloaders[domain], dataset_name=domain)
        # features_labels['target'] = compute_features(args, model_loaders[args.target] , dom_dataloaders[args.target], dataset_name=args.target)
        np_cls_centroids = compute_centroids(args,features_labels[domain]).cpu().numpy()
        np_feat = features_labels[domain]['feature'].cpu().numpy()
        # print('np_feat: ', np_feat.shape)
        np_label = features_labels[domain]['label'].cpu().numpy()
        # print('np_label:', np_label.shape)
        tsne_plotter(args, np_cls_centroids, np_feat, np_label, plt_name=domain)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clusformer')
    parser.add_argument('-b', '--batch_size', default=32, type=int,help='mini-batch size (default: 54)')
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('-s', '--source', type=str,help='Select the source [amazon, dslr, webcam]')
    parser.add_argument('-t', '--target', type=str,help='Select the target [amazon, dslr, webcam]')
    parser.add_argument('-d', '--dset', type=str,help='Select the target [amazon, dslr, webcam]')
    # parser.add_argument('-e', '--epochs', default=40, type=int,help='select number of cycles')
    parser.add_argument('-w', '--wandb', default=0, type=int,help='Log to wandb or not [0 - dont use | 1 - use]')
    parser.add_argument('-a', '--arch', default='rn50', type=str,help='Select student vit or rn50 based (default: rn50)')
    parser.add_argument('--net', default='vit', type=str,help='Select vit or rn50 based (default: vit)')
    parser.add_argument('-l', '--trained_wt', default='weights/office-home', type=str,help='Load src')
    args = parser.parse_args()
    print(args)
    main(args)
# data_size, dims, num_clusters = 1000, 2, 3
# x = np.random.randn(data_size, dims) / 6
# x = torch.from_numpy(x)

# # kmeans
# cluster_ids_x, cluster_centers = kmeans(
#     X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0')
# )
