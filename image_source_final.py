import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from wandb.sdk.lib import disabled
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import wandb
os.environ['WANDB_API_KEY'] = '93b09c048a71a2becc88791b28475f90622b0f63'
import sys
sys.path.append('..')
sys.path.append('common')
sys.path.append('src')

from data_helper import setup_datasets
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def attach_embd(prototypes, feats, dom_idx):
    dom_embd = prototypes[dom_idx] # accessing domain embedding according to domain index
                                    # shape: [bs, 512]
    x = torch.cat((feats, dom_embd), dim=1) # concat: ([bs, 2048], [bs, 512])
    return x

def cal_acc(loader, netF, netB, netC, prototypes, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0][0]
            labels = data[0][1]
            dom_idx = data[1]
            inputs = inputs.cuda()
            feats = netF(inputs)
            x = attach_embd(prototypes, feats, dom_idx)
            outputs = netC(netB(x))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc_oda(loader, netF, netB, netC, prototypes):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0][0]
            labels = data[0][1]
            dom_idx = data[1]
            inputs = inputs.cuda()
            feats = netF(inputs)
            x = attach_embd(prototypes, feats, dom_idx)
            outputs = netC(netB(x))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(args):
    # dset_loaders = data_load(args)
    dset_loaders = {}
    args.class_num, train_source_loader, train_target_loader, _, _ = setup_datasets(args)
    dset_loaders["source_tr"] = train_source_loader
    dset_loaders["source_te"] = train_source_loader
    dset_loaders["test"] = train_target_loader
    ## set base network

    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net,se=args.se,nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    
    ### test model paremet size
    # model=network.ResBase(res_name=args.net)
    # num_params = sum([np.prod(p.size()) for p in model.parameters()])
    # print("Total number of parameters: {}".format(num_params))
    # 
    # num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    # print("Total number of learning parameters: {}".format(num_params_update))

    args.feature_dim = netF.in_features
    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features + args.proto_dim,
        bottleneck_dim=args.bottleneck
    ).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    print("Loading Domian Embeddings from: %s" % args.proto_path)
    prototypes_file = osp.join(args.proto_path)
    prototypes = torch.load(prototypes_file)
    prototypes = torch.stack(list(prototypes.values()), dim=0)  # shape: [4, 512]
    prototypes = prototypes.cuda()

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            (inputs_source, labels_source), dom_idx = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            (inputs_source, labels_source), dom_idx = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        feats = netF(inputs_source) # [bs, 2048]

        # dom_embd = prototypes[dom_idx] # accessing domain embedding according to domain index
        #                                # shape: [bs, 512]
        # x = torch.cat((feats, dom_embd), dim=1) # concat: ([bs, 2048], [bs, 512])
        x = attach_embd(prototypes, feats, dom_idx)
        outputs_source = netC(netB(x))
        #print(args.class_num, outputs_source.shape, labels_source.shape)
        
        
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)


        optimizer.zero_grad()
        classifier_loss.backward()
        wandb.log({'SRC Train: train_classifier_loss': classifier_loss.item()})
        print(f'Task: {args.name_src}, Iter:{iter_num}/{max_iter} \t train_classifier_loss {classifier_loss.item()}')

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dataset=='visda-2017':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, prototypes, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, prototypes, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            wandb.log({'SRC TRAIN: Acc' : acc_s_te})
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:

                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

                torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
                print('Model Saved!!')

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
    test_target(args)
    print('Final Model Saved!!')

    return netF, netB, netC

def test_target(args):
    dset_loaders = {}
    args.class_num, train_source_loader, train_target_loader, _, _ = setup_datasets(args)
    dset_loaders["source_tr"] = train_source_loader
    dset_loaders["source_te"] = train_source_loader
    dset_loaders["test"] = train_target_loader

    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  
    else:
        netF = network.ViT().cuda()
        
    # args.feature_dim = netF.in_features
    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features + args.proto_dim,
        bottleneck_dim=args.bottleneck
    ).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    prototypes_file = osp.join(args.proto_path)
    prototypes = torch.load(prototypes_file)
    prototypes = torch.stack(list(prototypes.values()), dim=0)  # shape: [4, 512]
    prototypes = prototypes.cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC, prototypes)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dataset=='visda-2017':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, prototypes, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, prototypes, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SourceOnly Training')
    parser.add_argument('--source',default='Ar,Pr', type=str, help="source")
    parser.add_argument('--target',default='Cl,Rw', type=str, help="target")
    parser.add_argument('--root', default='data/', type=str, help="source")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--workers', type=int, default=8, help="number of workers")
    parser.add_argument('--dataset', type=str, default='OfficeHome', choices=['visda-2017', 'office', 'OfficeHome','pacs', 'domain_net'])
    parser.add_argument('--proto_path', help="path to Domain Embedding Prototypes")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--proto_dim', type=int, default=512)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='warmup')
    parser.add_argument('--da', type=str, default='oda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--bsp', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)
    parser.add_argument('--wandb', type=int, default=0)
    args = parser.parse_args()
    
    if args.dataset == 'OfficeHome':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        # args.class_num = 65 
    if args.dataset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        # args.class_num = 31
    if args.dataset == 'visda-2017':
        names = ['train', 'validation']
        # args.class_num = 12
    if args.dataset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        # args.class_num = 10
    if args.dataset == 'pacs':
        names = ['art_painting', 'cartoon', 'photo', 'sketch']
        # args.class_num = 7
    if args.dataset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch', 'real']
        # args.class_num = 345

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    mode = 'online' if args.wandb else 'disabled'
    wandb.init(project='degaa', entity='abd1', mode=mode)
    wandb.run.name =  'Warmup: {args.source}' + wandb.run.name
    print(print_args(args))

    args.output_dir_src = osp.join(args.output, args.da, args.dataset, args.source.replace(',',''))
    args.name_src = args.source.replace(',','')
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    # args.proto_path = "./protoruns/run7/prototypes.pth"
    assert osp.exists(args.proto_path), "Domain Embeddings Prototypes does not exists." 

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    args.name = args.source.replace(',','') +  '_' + args.target.replace(',','')
    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    
    train_source(args)
    test_target(args)
