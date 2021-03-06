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
import os
import sys
sys.path.append('..')
sys.path.append('common')
sys.path.append('src')
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
from data_helper import setup_datasets

sns.set(style="darkgrid")

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(args,dom_adapts): 

    model_loaders = {}
    # if args.dataset == 'office-home':
    #     wt_abbr = { 'Art': 'A','Clipart': 'C', 'Product': 'P', 'RealWorld': 'R'}

    # dom_adapts = wt_abbr[domain]

    print('Loading weights for ', dom_adapts)
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).to(device)
    elif args.net == 'vit':
        netF = network.ViT().to(device)
    
    args.feature_dim = netF.in_features
    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features + args.proto_dim,
        bottleneck_dim=args.bottleneck
     ).to(device)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device)

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


def attach_embd(prototypes, feats, dom_idx):
    dom_embd = prototypes[dom_idx] # accessing domain embedding according to domain index
                                    # shape: [bs, 512]
    x = torch.cat((feats, dom_embd), dim=1) # concat: ([bs, 2048], [bs, 512])
    return x


def compute_features(args, net, dataloader, prototypes, dataset_name=None):
    # print(len(dataloader))
    netF,netB, netC = net[0], net[1], net[2]
    print(f'Computing features for [{dataset_name}] -', len(dataloader)*args.batch_size, 'images')

    correct = 0
    total = 0
    stored_feat_lbl = {}
    stored_feat_lbl['feature'] = torch.empty((1,args.bottleneck)).to(device)
    stored_feat_lbl['label'] = torch.empty((1), dtype=torch.int).to(device)
    # print(store_feat)
    
    with torch.no_grad():
        iter_test = iter(dataloader)
        for i in tqdm(range(len(dataloader))):
            data = iter_test.next()
            inputs = data[0][0].to(device)
            labels = data[0][1].to(device)
            dom_idx = data[1]

            fout  = netF(inputs)
            x = attach_embd(prototypes, fout , dom_idx)
            feats = netB(x)
            
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
    
    run_sum = torch.zeros((args.class_num, args.bottleneck), dtype=torch.float).to(device)
    lbl_cnt = torch.ones((args.class_num), dtype=torch.float).to(device)
    cls_centroids = torch.zeros((args.class_num, args.bottleneck), dtype=torch.float).to(device)

    for feat, lbl in zip(features_labels['feature'],features_labels['label']):
        run_sum[lbl] += feat
        lbl_cnt[lbl] += 1
        
    for i in range(args.class_num):
        # if lbl_cnt[i] == 0:
        #     print('Failed computing centroid (no images present) for Class', i)
        cls_centroids[i] = run_sum[i] / lbl_cnt[i]
    return cls_centroids
    # print(run_sum.shape)
    
def tsne_plotter(args, np_cls_centroids, np_feat, np_label, plt_name=None, save_cnt_path=None):

        cat_feat = np.concatenate((np_cls_centroids, np_feat), axis=0)
        save_folder = f'{save_cnt_path}/{args.dataset}'
        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.save(f'{save_folder}/{plt_name}_centroid.npy', np_cls_centroids)
            print('Centroid Saved')

        data_cen, data_feat = {}, {}
        X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(cat_feat)
        data_cen['x_cent'], data_cen['y_cent'], data_cen['lbl_cent'] = X_embedded[:args.class_num,0],X_embedded[:args.class_num,1], np.arange(args.class_num)
        data_feat['x_feat'], data_feat['y_feat'], data_feat['lbl_feat'] = X_embedded[args.class_num:,0],X_embedded[args.class_num:,1], np_label

        data_feat=pd.DataFrame(data_feat)
        data_cen=pd.DataFrame(data_cen)

        sns.scatterplot(data=data_feat, x="x_feat", y="y_feat", legend=False, hue='lbl_feat', 
                            palette=sns.color_palette("Spectral", as_cmap=True) , alpha=0.5, s=12)
        sns.scatterplot(data=data_cen, x="x_cent", y="y_cent", hue='lbl_cent', legend=False, 
                            marker='X', palette=sns.color_palette("Spectral", as_cmap=True))
        
        if not os.path.exists('tsne_plts'):
            os.makedirs('tsne_plts')                    
        plt.savefig(f'tsne_plts/tsne_{plt_name}.png')

def main(args):

    dom_dataloaders = {}
    # dom_dataloaders, dataset_loaders_test = data_load(args, batch_size=args.batch_size) 
    args.class_num, train_src_loader, train_target_loader, _, _ = setup_datasets(args)
    args.source = ''.join(args.source)
    args.target = ''.join(args.target)
    dom_dataloaders[args.source] = train_src_loader
    dom_dataloaders[args.target] = train_target_loader
    model_loaders ={}

    print("Loading Domian Embeddings from: %s" % args.proto_path)
    prototypes_file = osp.join(args.proto_path)
    prototypes = torch.load(prototypes_file)
    prototypes = torch.stack(list(prototypes.values()), dim=0)  # shape: [4, 512]
    prototypes = prototypes.to(device)

    for domain in [args.source, args.target]:
        model_loaders[domain] = load_models(args, domain)
        features_labels = {}

        # Compute features and labels of src and targets using src trained wts
        features_labels[domain] = compute_features(args, model_loaders[domain] , dom_dataloaders[domain], prototypes, dataset_name=domain)
        # features_labels['target'] = compute_features(args, model_loaders[args.target] , dom_dataloaders[args.target], dataset_name=args.target)
        np_cls_centroids = compute_centroids(args,features_labels[domain]).cpu().numpy()
        np_feat = features_labels[domain]['feature'].cpu().numpy()
        # print('np_feat: ', np_feat.shape)
        np_label = features_labels[domain]['label'].cpu().numpy()
        # print('np_label:', np_label.shape)
        tsne_plotter(args, np_cls_centroids, np_feat, np_label, plt_name=domain, save_cnt_path='centroids')
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Centroids')
    parser.add_argument('-b', '--batch_size', default=64, type=int,help='mini-batch size (default: 64)')
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--root', default='data/', type=str, )
    parser.add_argument('--workers', default=8, type=int )
    parser.add_argument('-s', '--source',default='Ar,Pr', type=str,help='Select the source [amazon, dslr, webcam]')
    parser.add_argument('-t', '--target', default='Cl,Rw',type=str,help='Select the target [amazon, dslr, webcam]')
    parser.add_argument('-d', '--dataset',default='OfficeHome', type=str,help='Select the target [amazon, dslr, webcam]')
    parser.add_argument('--proto_path', help="path to Domain Embedding Prototypes")
    # parser.add_argument('-e', '--epochs', default=40, type=int,help='select number of cycles')
    parser.add_argument('--proto_dim', type=int, default=512)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('-w', '--wandb', default=0, type=int,help='Log to wandb or not [0 - dont use | 1 - use]')
    parser.add_argument('--net', default='resnet50', type=str,help='Select vit or resnet50 based (default: resnet50)')
    parser.add_argument('-l', '--trained_wt', default='weights/uda/OfficeHome', type=str,help='Load src')
    args = parser.parse_args()
    print(args)


    # args.proto_path = "./protoruns/run7/prototypes.pth"
    assert osp.exists(args.proto_path), "Domain Embeddings Prototypes does not exists." 

    main(args)
# data_size, dims, num_clusters = 1000, 2, 3
# x = np.random.randn(data_size, dims) / 6
# x = torch.from_numpy(x)

# # kmeans
# cluster_ids_x, cluster_centers = kmeans(
#     X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0')
# )
