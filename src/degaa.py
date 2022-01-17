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
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import numpy as np


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

import wandb

warnings.filterwarnings("ignore")

# wandb.init(project = "degaa")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Total GPUs Used:", torch.cuda.device_count())
i = 0
print("Hardwares Used: ")
while i < torch.cuda.device_count():
    print(torch.cuda.get_device_name(i))
    i = i + 1


counter = 0

def main(args: argparse.Namespace):
    if args.wandb:
        mode = "online" if args.wandb else "disabled"
        wandb.init(project="degaa", entity="flagarihant2000", mode=mode)
        wandb.run.name = "run_" + str(args.source) + str(args.target)
    if args.tensorboard:
        global writer
        writer = SummaryWriter(os.path.join(args.output_dir,"tensorboard"))

    logger = CompleteLogger(args.log, args.phase)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        # warnings.warn(
        #     "You have chosen to seed training. "
        #     "This will turn on the CUDNN deterministic setting, "
        #     "which can slow down your training considerably! "
        #     "You may see unexpected behavior when restarting "
        #     "from checkpoints."
        # )

        cudnn.benchmark = True

    '''
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose(
        [
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )

    val_transform = T.Compose(
        [ResizeImage(256), T.CenterCrop(224), T.ToTensor(), normalize]
    )

    dataset = datasets.__dict__[args.dataset]
    source_dataset = open_set(dataset, source=True)
    target_dataset = open_set(dataset, source=False)
    # source_dataset = dataset
    # target_dataset = dataset
    # OfficeHome Ar: Art, Cl: Clipart, Pr: Product, Rw: Real - World
    # print(args.source, args.target, dataset.domains())
    # print(args.source.split(','))
    args.source = args.source.split(",")
    args.target = args.target.split(",")
    if args.dataset == "OfficeHome":
        args.root = os.path.join(args.root, "office-home")
    train_source_dataset = ConcatDataset(
        [
            source_dataset(
                root=args.root, task=source, download=True, transform=train_transform
            )
            for source in args.source
        ]
    )
    train_target_dataset = ConcatDataset(
        [
            target_dataset(
                root=args.root, task=target, download=True, transform=train_transform
            )
            for target in args.target
        ]
    )
    # train_source_dataset = source_dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    # train_target_dataset = target_dataset(root=args.root, task=args.target, download=True, transform=train_transform)

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    # val_dataset = target_dataset(
    #     root=args.root, task=args.target[0], download=True, transform=val_transform
    # )
    val_dataset = ConcatDataset(
        [
            target_dataset(
                root=args.root, task=target, download=True, transform=train_transform
            )
            for target in args.target
        ]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    ) # change shuffle to Flase later

    if args.dataset == "DomainNet":
        test_dataset = dataset(
            root=args.root,
            task=args.target,
            split="test",
            download=True,
            transform=val_transform,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
    else:
        test_loader = val_loader
    '''
    # val_iter = ForeverDataIterator(val_loader)

    num_classes, train_source_loader, train_target_loader, val_loader, test_loader = setup_datasets(args, concat=True)
    # num_classes = num_classes - 1
    print(f"Num Classes: {num_classes}")
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # backbone = models.__dict__[args.arch](pretrained = True)
    netF = network.ResBase(res_name=args.net).to(device)
    modelpathB = "weights/uda/office-home/A/source_B.pt"

    # num_classes = args.num_classes  # train_source_dataset.num_classes
    # num_classes = len(train_source_dataset.datasets[0].classes)
    args.num_classes = num_classes

    # classifier = ImageClassifier(backbone, num_classes = num_classes, bottleneck_dim = args.bottleneck_dim).to(device)
    # summary(classifier, (3, 224, 224))

    args.feature_dim = netF.in_features
    args.bottleneck_dim = 256

    proto_dim = 512  # to be chznaged
    netB = network.feat_bootleneck(
        type="bn",
        feature_dim=args.feature_dim + proto_dim, # 2048 + 512
        bottleneck_dim=args.bottleneck_dim, # 256
    ).to(device)
    netC = network.feat_classifier(
        type="wn", class_num=args.num_classes, bottleneck_dim=args.bottleneck_dim  # 256, #26
    ).to(device)

    modelpathF = f'{args.trained_wt}/{args.dataset}/{"".join(args.source)}/source_F.pt'
    netF.load_state_dict(torch.load(modelpathF))
    
    # modelpathB = f'{args.trained_wt}/{args.dataset}/{"".join(args.source)}/source_B.pt'
    # netB.load_state_dict(torch.load(modelpathB))

    modelpathC = f'{args.trained_wt}/{args.dataset}/{"".join(args.source)}/source_C.pt'
    netC.load_state_dict(torch.load(modelpathC))

    gaa = GAA(
        input_dim=args.bottleneck_dim,
        num_classes=num_classes,
        gnn_layers=6,
        num_heads=4,
    ).to(device)

    classifier = [netF, netB, netC]

    # optimizer = SGD(classifier.get_parameters(), args.lr, momentum = args.momentum, weight_decay = args.wd, nesterov = True)
    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}] 
    for k, v in gaa.named_parameters():
        param_group += [{"params": v, "lr": learning_rate}]
    optimizer = SGD(param_group)
    lr_scheduler = LambdaLR(
        optimizer,
        lambda x: args.lr * (1 + args.lr_gamma * float(x)) ** (-args.lr_decay),
    )


    # summary(gaa, [(32, 1024),(32, 1024)])
    centroids = np.load('/home/vikash/project/rohit_lal/os-nsmt/centroids/OfficeHome/ArPr_centroid.npy')
    centroids = centroids[:num_classes-1]
    # print(centroids.shape)

    prototypes_file = os.path.join("/home/vikash/project/rohit_lal/os-nsmt/protoruns/run4/prototypes.pth")
    prototypes = torch.load(prototypes_file)
    # print(prototypes.keys())
    prototypes = torch.stack(list(prototypes.values()), dim=0)  # shape: [4, 512]
    prototypes = prototypes.to(device)

    if args.phase == "test":
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    best_h_score = 0.0
    for epoch in range(args.epochs):
        train(
            train_source_iter,
            train_target_iter,
            classifier,
            gaa,
            centroids,
            prototypes,
            optimizer,
            lr_scheduler,
            epoch,
            args,
            num_classes
        )
        h_score = validate(val_loader, classifier, prototypes, args)

        if h_score > best_h_score:
            # torch.save(classifier, logger.get_checkpoint_path("latest"))             
            torch.save(netF, osp.join(args.output_dir, "latest_source_F.pt"))
            torch.save(netB, osp.join(args.output_dir, "latest_source_B.pt"))
            torch.save(netC, osp.join(args.output_dir, "latest_source_C.pt"))
            # shutil.copy(
            #     logger.get_checkpoint_path("latest"), logger.get_checkpoint_path("best")
            # )

    print("best_h_score = {:3.1f}".format(best_h_score))

    torch.save(netF, osp.join(args.output_dir, "final_source_F.pt"))
    torch.save(netB, osp.join(args.output_dir, "final_source_B.pt"))
    torch.save(netC, osp.join(args.output_dir, "final_source_C.pt"))

    # classifier.load_state_dict(torch.load(logger.get_checkpoint_path("best")))
    h_score = validate(test_loader, classifier, prototypes, args)
    print("test_h_score = {:3.1f}".format(h_score))

    logger.close()
    writer.close()


def NearestNeighbor(known, centroids):
    dist = torch.cdist(known, centroids.to(torch.float32), p=2)
    knn = dist.topk(1, largest=False, dim=1) # dim=1 to access dominant in each of these inlier vectors
    return knn.indices


def train(
    train_source_iter: ForeverDataIterator,
    train_target_iter: ForeverDataIterator,
    model: ImageClassifier,
    gaa: GAA,
    centroids,
    prototypes,
    optimizer: SGD,
    lr_scheduler: LambdaLR,
    epoch: int,
    args: argparse.Namespace,
    num_classes=65,
):

    global counter

    batch_time = AverageMeter("Time", ":5.2f")
    data_time = AverageMeter("Data", ":5.2f")
    losses = AverageMeter("Loss", ":6.2f")
    cls_accs = AverageMeter("Cls Acc", ":3.1f")
    tgt_accs = AverageMeter("Tgt Acc", ":3.1f")
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, tgt_accs],
        prefix="Epoch: [{}]".format(epoch),
    )

    netF, netB, netC = model[0], model[1], model[2]
    netF.train()
    netB.train()
    netC.train()
    gaa.train()
    clf = LocalOutlierFactor(n_neighbors=num_classes + 1, contamination=0.1)
    softmax = nn.Softmax(dim=1)

    end = time.time()
    for i in range(args.iters_per_epoch):
        (x_s, label_s), ds_idx = next(train_source_iter)
        (x_t, label_t), dt_idx = next(train_target_iter)

        x_s = x_s.to(device)
        label_s = label_s.to(device)
        x_t = x_t.to(device)
        label_t = label_t.to(device)
        

        data_time.update(time.time() - end)
        # x_s : Art + Webcam
        # torch.where(embd, idx=doimain_indxs) # shape: [bs*num_domains, 512]
        ds_embd = prototypes[ds_idx]  # shape: [bs, 512]
        dt_embd = prototypes[dt_idx]
        d_embd = torch.cat((ds_embd, dt_embd), dim=0)
        x = torch.cat((x_s, x_t), dim=0)  # x.shape [bs*2, 3, 224]
        # embd.shape [bs*2, 512]
        # embd[:bs] = embd{1},
        # f = netB(netF(x)) # Seperate and contact netF(x) with embedding along dim 1
        feats = netF(x)
        f = torch.cat([feats, d_embd], dim=1)  # [bs, 2048+512]
        f = netB(f)

        # y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        label_t = torch.empty_like(label_t) # shape [bs]
        f_t_numpy = f_t.clone().cpu().detach().numpy()
        y_pred = clf.fit_predict(f_t_numpy)
        index = np.where(y_pred == -1)  # for n outliers, shape [n, ..]
        label_t[index] = num_classes - 1 # 25 for all outliers !!
        # [.., .., ..., 25, .., 25, ..,  25, ..]

        index1 = np.where(y_pred == 1)  # for known classes [bs-n, ...]

        """
    get centroids and find classes.
    """
        # bottle_s = bottle(f_s)
        # bottle_t = bottle(f_t)
        known = f_t[index1]
        if not isinstance(centroids, torch.Tensor):
            centroids = torch.from_numpy(centroids).to(device)
        known_idx = NearestNeighbor(known, centroids)
        label_t[index1] = known_idx.squeeze()  # asigning target domain classes on bsis of KNN
                                                # values from 0 to 24 for all known classes 
                                                # # [5, 6, 0, 25, 24, 25, 3, 25, 6]

        f_s, f_t = gaa(f_s, f_t)
        y_s, y_t = netC(f_s), netC(f_t)
        y_s, y_t = softmax(y_s), softmax(y_t)
        # print(label_t)
        cls_loss_s = F.cross_entropy(y_s, label_s)
        cls_loss_t = F.cross_entropy(y_t, label_t)
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

        counter = counter + 1
        if args.wandb:
            wandb.log(
                {"accuracy_source": cls_acc, "accuracy_target": tgt_acc, "loss": loss}
            )
        if args.tensorboard:
            writer.add_scalar('accuracy_source', cls_acc, counter)
            writer.add_scalar('accuracy_target', cls_acc, counter)
            writer.add_scalar('loss', loss, counter)

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: Classifier, prototypes, args: argparse.Namespace):
    batch_time = AverageMeter("Time", ":6.3f")
    classes = val_loader.dataset.datasets[0].classes
    confmat = ConfusionMatrix(len(classes))
    progress = ProgressMeter(len(val_loader), [batch_time], prefix="Test: ")
    netF, netB, netC = model[0], model[1], model[2]
    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with torch.no_grad():
        end = time.time()
        for i, ((images, target), d_idx) in enumerate(val_loader):
            images = images.to(device)
            # target = target.to(device)

            # compute output
            # output, _ = model(images)
            d_embd = prototypes[d_idx]
            feats = netF(images)
            x = torch.cat([feats, d_embd], dim=1) # Concat Features with Domain embeddings
            output = netC(netB(x))

            softmax_output = F.softmax(output, dim=1)
            # softmax_output[:, -1] = args.threshold  # look into this

            _, preds= torch.max(softmax_output, 1)
            if start_test:
                all_output = preds.float().cpu()
                all_label = target.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, preds.float().cpu()), 0)
                all_label = torch.cat((all_label, target.float().cpu()), 0)



            # measure accuracy and record loss
            confmat.update(target, softmax_output.argmax(1).cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            


        matrix = confusion_matrix(all_label, all_output)
        accs = matrix.diagonal()/matrix.sum(axis=1)
        print(accs)

        # acc_global, accs, iu = confmat.compute()
        all_acc = np.mean(accs).item() * 100
        known = np.mean(accs[:-1]).item() * 100
        unknown = accs[-1].item() * 100
        h_score = 2 * np.divide(np.multiply(known, unknown), np.add(known, unknown))
        # h_score = 2 * known * unknown / (known + unknown)
        if args.per_class_eval:
            print(confmat.format(classes))
        print(
            " * All {all:.3f} Known {known:.3f} Unknown {unknown:.3f} H-score {h_score:.3f}".format(
                all=all_acc, known=known, unknown=unknown, h_score=h_score
            )
        )

    return h_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DEGAA")

    parser.add_argument("-d", "--dataset", default="OfficeHome")
    parser.add_argument("-a", "--arch", default="resnet50")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("--lr", "--learning_rate", default=0.002)
    parser.add_argument("--lr-gamma", default=0.001)
    parser.add_argument("--bottleneck-dim", default=256)
    parser.add_argument("--feature-dim", default=256)
    parser.add_argument("--lr-decay", default=0.75)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--wd", "--weight-decay", default=1e-3)
    parser.add_argument("-j", "--workers", default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--root", default="./data")
    parser.add_argument("-s", "--source", help="source domain(s)", default="Ar,Pr")
    parser.add_argument("-t", "--target", help="target domain(s)", default="Cl,Rw")
    parser.add_argument("--trade-off", default=1.0, type=float)
    parser.add_argument("-i", "--iters-per-epoch", default=500, type=int)
    parser.add_argument("-p", "--print-freq", default=100, type=int)
    parser.add_argument("--log", type=str, default="degaa")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--threshold", default=0.8, type=float)
    parser.add_argument("--per-class-eval", action="store_true")
    parser.add_argument("--tensorboard", action='store_true', help='enables tensorboard logging')
    parser.add_argument("--wandb", action="store_true", help="enables wandb logging")
    parser.add_argument("--output_dir", default="./adapt/run1", help="enables wandb logging")
    parser.add_argument('-l', '--trained_wt', default='weights/uda', type=str,help='Load src')
    parser.add_argument(
        "--net",
        default="resnet50",
        type=str,
        help="Select vit or rn50 based (default: vit)",
    )
    # parser.add_argument("-n", "--num_classes", default=26)

    args = parser.parse_args()
    main(args)
