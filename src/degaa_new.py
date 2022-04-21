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
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("..")
sys.path.append("")
from common.modules.classifier import Classifier

from dalib.adaptation.degaa import ImageClassifier, GAA
import common.vision.datasets.openset as datasets
from common.vision.datasets.openset import default_open_set as open_set
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger, TextLogger
from common.utils.analysis import collect_feature, tsne, a_distance
import network
from data_helper import setup_datasets
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

import wandb
os.environ['WANDB_API_KEY'] = '93b09c048a71a2becc88791b28475f90622b0f63'

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Total GPUs Used:", torch.cuda.device_count())
i = 0
print("Hardwares Used: ")
while i < torch.cuda.device_count():
    print(torch.cuda.get_device_name(i))
    i = i + 1


counter = 0


def attach_embd(prototypes, feats, dom_idx):
    dom_embd = prototypes[dom_idx] # accessing domain embedding according to domain index
                                    # shape: [bs, 512]
    x = torch.cat((feats, dom_embd), dim=1) # concat: ([bs, 2048], [bs, 512])
    return x


def main(args: argparse.Namespace):
    if args.wandb:
        mode = "online" if args.wandb else "disabled"
        wandb.init(project="degaa", entity="abd1", mode=mode)
        wandb.run.name = wandb.run.name + f" S: {str(args.source)} T: {str(args.target)}"
    if args.tensorboard:
        global writer
        writer = SummaryWriter(os.path.join(args.output_dir,"tensorboard"))

    # logger = CompleteLogger(args.log, args.phase)
    global logger
    logger = TextLogger(osp.join(args.output_dir, "out.txt"))

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


    global val_loader
    num_classes, train_source_loader, train_target_loader, val_loader, test_loader = setup_datasets(args, concat=True)
    # num_classes = num_classes - 1
    print(f"Num Classes: {num_classes}")
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # backbone = models.__dict__[args.arch](pretrained = True)
    netF = network.ResBase(res_name=args.net).to(device)

    # num_classes = args.num_classes  # train_source_dataset.num_classes
    # num_classes = len(train_source_dataset.datasets[0].classes)
    args.num_classes = num_classes

    # classifier = ImageClassifier(backbone, num_classes = num_classes, bottleneck = args.bottleneck).to(device)
    # summary(classifier, (3, 224, 224))

    args.feature_dim = netF.in_features
    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=args.feature_dim + args.proto_dim, # 2048 + 512
        bottleneck_dim=args.bottleneck, # 256
    ).to(device)
    netC = network.feat_classifier(
        type=args.layer, class_num=args.num_classes, bottleneck_dim=args.bottleneck  # 256, #26
    ).to(device)

    modelpathF = f'{args.trained_wt}/{args.dataset}/{"".join(args.source)}/source_F.pt'
    print("Loading netF from", modelpathF)
    netF.load_state_dict(torch.load(modelpathF))
    
    modelpathB = f'{args.trained_wt}/{args.dataset}/{"".join(args.source)}/source_B.pt'
    print("Loading netB from", modelpathB)
    netB.load_state_dict(torch.load(modelpathB))

    modelpathC = f'{args.trained_wt}/{args.dataset}/{"".join(args.source)}/source_C.pt'
    print("Loading netC from", modelpathC)
    netC.load_state_dict(torch.load(modelpathC))

    gaa = GAA(
        input_dim=args.bottleneck,
        num_classes=num_classes,
        gnn_layers=6,
        num_heads=4,
    ).to(device)

    global classifier
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
    optimizer = SGD(param_group, momentum = args.momentum, weight_decay = args.wd, nesterov = True)
    lr_scheduler = LambdaLR(
        optimizer,
        lambda x: args.lr * (1 + args.lr_gamma * float(x)) ** (-args.lr_decay),
    )


    # summary(gaa, [(32, 1024),(32, 1024)])
    centroids = np.load(args.centroid_path)
    centroids = centroids[:num_classes-1]
    centroids = torch.from_numpy(centroids).to(device)
    print("Loading centroids from", args.centroid_path)
    # print(centroids.shape)

    prototypes_file = os.path.join(args.proto_path)
    prototypes = torch.load(prototypes_file)
    # print(prototypes.keys())
    prototypes = torch.stack(list(prototypes.values()), dim=0)  # shape: [4, 512]
    prototypes = prototypes.to(device)

    if args.phase == "test":
        acc = validate(val_loader, classifier, prototypes, args)
        print("val_acc = {:3.1f}".format(acc))
        return

    train(
        train_source_iter,
        train_target_iter,
        classifier,
        gaa,
        centroids,
        prototypes,
        optimizer,
        lr_scheduler,
        # epoch,
        args,
        num_classes
    )

    torch.save(netF.state_dict(), osp.join(args.output_dir, "final_source_F.pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "final_source_B.pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "final_source_C.pt"))

    acc = validate(val_loader, classifier, prototypes, args)
    print("val_acc = {:3.1f}".format(acc))

    logger.close()
    if args.tensorboard:
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
    # epoch: int,
    args: argparse.Namespace,
    num_classes=65,
):
    best_acc = 0.0
    # F_S, F_T, Label_S, Label_T = [], [], [], []
    Temp_DataLoader = None
    # Psuedo_Labels = torch.empty(train_target_iter.data_loader.dataset.__len__())
    Psuedo_Labels = np.empty(train_target_iter.data_loader.dataset.__len__())
    for epoch in range(args.epochs):

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
        clf = LocalOutlierFactor(n_neighbors=2000,  contamination=0.4, metric = "cosine")
        softmax = nn.Softmax(dim=1)

        # args.iters_per_epoch = int(8704/args.batch_size)
        args.iters_per_epoch = len(train_target_iter)
        end = time.time()

        if epoch == 0 or epoch % args.episodes == 0:
            with torch.no_grad():
                F_S, F_T, Label_S, Label_T = [], [], [], []
                T_idxs = []
                for i in tqdm(range(args.iters_per_epoch), "LOF and Pseudo labels"):
                    # (x_s, label_s), ds_idx, s_idx = next(train_source_iter)
                    (x_t, label_t), dt_idx, t_idx = next(train_target_iter)
                    # x_s = x_s.to(device)
                    x_t = x_t.to(device)

                    feat_t = netF(x_t)
                    f_t = attach_embd(prototypes, feat_t, dt_idx)
                    f_t = netB(f_t)
                    f_t = F.normalize(f_t, dim=0)
                    
                    # x = torch.cat((x_s, x_t), dim=0)
                    # dom_idx = torch.cat((ds_idx, dt_idx), dim=0)
                    # feats = netF(x)
                    # f = attach_embd(prototypes, feats, dom_idx)
                    # f = netB(f)
                    # f = F.normalize(f, dim=0)
                    # f_s, f_t = f.chunk(2, dim=0)

                    # F_S.append(f_s)
                    # Label_S.append(label_s)
                    Label_T.append(label_t.cpu().detach())
                    F_T.append(f_t.cpu().detach().numpy())
                    T_idxs.append(t_idx)

                # F_S = torch.cat(F_S)
                # Label_S = torch.cat(Label_S)
                Label_T = torch.cat(Label_T)
                # F_T = torch.cat(F_T)

                # S_idxs = torch.cat(S_idxs)
                T_idxs = torch.cat(T_idxs)

                F_T = np.concatenate(F_T)
                Y_Preds = clf.fit_predict(F_T)
                unknown_acc = np.intersect1d(np.where(Y_Preds==-1), np.where(Label_T==25)).shape[0] / sum(Label_T==25)
                logger.write(f"EPOCH: {epoch:3.0f} LOF Unknown accuracy (num_correct_predicted_outliers / num_total_outliers): {unknown_acc:.3f}\n")
                logger.flush()
                
                Label_TT = torch.empty_like(Label_T)
                # index = np.where(Y_Preds == -1)[0]
                index = (Y_Preds == -1)
                Label_TT[index] = num_classes - 1 # label=25 where y_pred=-1

                Known = F_T[~index]
                Known_idx = NearestNeighbor(torch.from_numpy(Known).to(device), centroids).cpu()
                Label_TT[~index] = Known_idx.squeeze()

                psuedo_label_acc = Label_TT[~index].eq(Label_T[~index]).sum() / Label_T[~index].shape[0]
                logger.write(f"EPOCH: {epoch:3.0f} Pseudo Label accuracy: {psuedo_label_acc:.3f}")
                logger.flush()
                print(f"EPOCH: {epoch:3.0f} Pseudo Label accuracy: {psuedo_label_acc:.3f}")

                if args.wandb:
                    wandb.log(
                        {"LOF_accuracy": unknown_acc , "Pseudo_label_accuracy": psuedo_label_acc}
                    )

                Psuedo_Labels[T_idxs] = Label_TT
                if type(Psuedo_Labels) is np.ndarray:
                    Psuedo_Labels = torch.from_numpy(Psuedo_Labels).long()
                del F_T
                # F_T = torch.from_numpy(F_T)
                # Temp_DataLoader = DataLoader(TensorDataset(F_S, Label_S, F_T, Label_TT), args.batch_size, shuffle=True, num_workers=args.workers)


        for i in tqdm(range(args.iters_per_epoch), "GAA"):
            (x_s, label_s), ds_idx, s_idx = next(train_source_iter)
            (x_t, label_t), dt_idx, t_idx = next(train_target_iter)

            x_s = x_s.to(device)
            label_s = label_s.to(device)
            x_t = x_t.to(device)
            label_t = label_t.to(device)
            label_tt = Psuedo_Labels[t_idx].to(device)

            x = torch.cat((x_s, x_t), dim=0)
            dom_idx = torch.cat((ds_idx, dt_idx), dim=0)
            feats = netF(x)
            # print(netF.conv1.weight[0,0,0,0].item(), netF.conv1.weight[0,0,1,1].item(), netF.conv1.weight[2,1,0,0].item())
            f = attach_embd(prototypes, feats, dom_idx)
            f = netB(f)
            f_s, f_t = f.chunk(2, dim=0)

            f_s, f_t = gaa(f_s, f_t)
            y_s, y_t = netC(f_s), netC(f_t)
            y_s, y_t = softmax(y_s), softmax(y_t)

            cls_loss_s = F.cross_entropy(y_s, label_s)
            cls_loss_t = F.cross_entropy(y_t, label_tt)
            loss = cls_loss_s + cls_loss_t * args.trade_off


            cls_acc = accuracy(y_s, label_s)[0]
            tgt_acc = accuracy(y_t, label_tt)[0]

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
                    {"Accuracy_Source": cls_acc, "Accuracy_Target(Train)": tgt_acc, "loss": loss}
                )
            if args.tensorboard:
                writer.add_scalar('accuracy_source', cls_acc, counter)
                writer.add_scalar('accuracy_target', cls_acc, counter)
                writer.add_scalar('loss', loss, counter)

            # if i % args.print_freq == 0:
            #     progress.display(i)



            '''
            # x_s : Art + Webcam
            # torch.where(embd, idx=doimain_indxs) # shape: [bs*num_domains, 512]
            # ds_embd = prototypes[ds_idx]  # shape: [bs, 512]
            # dt_embd = prototypes[dt_idx]
            # d_embd = torch.cat((ds_embd, dt_embd), dim=0)
            # embd.shape [bs*2, 512]
            # embd[:bs] = embd{1},
            # f = netB(netF(x)) # Seperate and contact netF(x) with embedding along dim 1
            feats = netF(x)
            f = attach_embd(prototypes, feats, dom_idx) # [bs*2, 2048+512]
            # f = torch.cat([feats, d_embd], dim=1)  # [bs*2, 2048+512]
            f = netB(f)

            # y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)
            '''
                



            # label_s = torch.empty_like(label_s) # shape [bs]
            # f_s_numpy = f_s.clone().cpu().detach().numpy()
            # source_pred= clf.fit_predict(f_s_numpy) # 4. pass whole dataset through LOF
            # print(label_s, "\n", source_pred)
            # 1. Try training without Lof and wothout unknown classes. Using target labels expracted from centroids

        if epoch == 0 or epoch % args.episodes == 0:
            acc = validate(val_loader, classifier, prototypes, args)

            if acc > best_acc:
                best_acc = acc

                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

                torch.save(best_netF, osp.join(args.output_dir, "best_source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir, "best_source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir, "best_source_C.pt"))

    print("best_acc = {:3.1f}".format(acc))


# chamge it to cal_acc_oda
def validate(val_loader: DataLoader, model: Classifier, prototypes, args: argparse.Namespace):
    batch_time = AverageMeter("Time", ":6.3f")
    classes = val_loader.dataset.datasets[0].classes
    print("Num Classes:", len(classes))
    # confmat = ConfusionMatrix(len(classes))
    progress = ProgressMeter(len(val_loader), [batch_time], prefix="Test: ")
    netF, netB, netC = model[0], model[1], model[2]
    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with torch.no_grad():
        end = time.time()
        for i, ((images, target), d_idx, _) in enumerate(val_loader):
            images = images.to(device)
            # target = target.to(device)

            # compute output
            # output, _ = model(images)
            feats = netF(images)
            x = attach_embd(prototypes, feats, d_idx)
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
            # confmat.update(target, softmax_output.argmax(1).cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            
        matrix = confusion_matrix(all_label, all_output)
        # import seaborn as sns
        # sns.heatmap(matrix)
        # plt.savefig('cmt_adapt')

        accs = matrix.diagonal()/matrix.sum(axis=1) * 100
        acc = matrix.diagonal().sum() / matrix.sum() * 100

        known_acc = matrix[:-1, :-1].diagonal().sum() / matrix[:-1, :-1].sum() * 100
        unknown_acc = accs[-1] * 100
        # print(accs)

        # acc_global, accs, iu = confmat.compute()
        # all_acc = np.mean(accs).item() * 100
        # known = np.mean(accs[:-1]).item() * 100
        # unknown = accs[-1].item() * 100
        # h_score = 2 * np.divide(np.multiply(known, unknown), np.add(known, unknown))
        # h_score = 2 * known * unknown / (known + unknown)
        # if args.per_class_eval:
        #     print(confmat.format(classes))
        # print(
        #     " * All {all:.3f} Known {known:.3f} Unknown {unknown:.3f} H-score {h_score:.3f}".format(
        #         all=all_acc, known=known, unknown=unknown, h_score=h_score
        #     )
        # )
        print("Accuracy {:.3f}, Known {:.3f}, Unknown {:.3f}".format(acc, known_acc, unknown_acc))
        if args.wandb:
            # cm = wandb.plot.confusion_matrix(y_true=all_label.numpy(), preds=all_output.numpy())
            # wandb.log({"conf_mat": cm})
            wandb.log(
                {"Accuracy_Target(Val)": acc, "Accuracy_Known": known_acc, "Accuracy_Unknown": unknown_acc}
            )

    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DEGAA")

    parser.add_argument("-d", "--dataset", default="OfficeHome")
    # parser.add_argument("-a",c "--arch", default="resnet50")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("--lr", "--learning_rate", default=0.01)
    parser.add_argument("--lr-gamma", default=0.001)
    parser.add_argument("--bottleneck-dim", default=256)
    parser.add_argument("--feature-dim", default=256)
    parser.add_argument("--lr-decay", default=0.75)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--wd", "--weight-decay", default=1e-3)
    parser.add_argument("-j", "--workers", default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--root", default="./data")
    parser.add_argument("-s", "--source", help="source domain(s)", default="Ar,Pr")
    parser.add_argument("-t", "--target", help="target domain(s)", default="Cl,Rw")
    parser.add_argument('-p', '--proto_path', help="path to Domain Embedding Prototypes")
    parser.add_argument('-c', '--centroid_path', help="path to Source Only trained Centroids")
    parser.add_argument('--proto_dim', type=int, default=512)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--trade-off", default=1.0, type=float)
    parser.add_argument("-i", "--iters-per-epoch", default=1, type=int)
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--log", type=str, default="degaa")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--threshold", default=0.8, type=float)
    parser.add_argument("--per-class-eval", action="store_true")
    parser.add_argument("--tensorboard", action='store_true', help='enables tensorboard logging')
    parser.add_argument("--wandb", action="store_true", help="enables wandb logging")
    parser.add_argument("--output_dir", default="./adapt/run4", help="enables wandb logging")
    parser.add_argument('-l', '--trained_wt', default='weights_final/oda', type=str,help='Load src')
    parser.add_argument(
        "--net",
        default="resnet50",
        type=str,
        help="Select vit or rn50 based (default: resnet50)",
    )
    # parser.add_argument("-n", "--num_classes", default=26)

    args = parser.parse_args()

    args.proto_path = "./protoruns/run7/prototypes.pth"
    assert osp.exists(args.proto_path), "Domain Embeddings Prototypes does not exists." 

    args.centroid_path =  f'./centroids/{args.dataset}/{"".join(args.source)}_centroid.npy'
    assert osp.exists(args.centroid_path), "Source Only trained centroids does not exists." 

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
