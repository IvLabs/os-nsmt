#!/usr/bin/env python3

import random

import torch
import torch.nn as nn
from . import networks
from .lib.misc import cross_entropy, random_pairs_of_minibatches
from .lib.prototype import prototypical_loss


ALGORITHMS_Proto = ["Proto", "Proto_NoReLU", "Proto_Mixup", "Proto_NoReLU_Mixup"]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Proto(nn.Module):
    """
    Domain-Aware Prototypical Domain Generalization (Proto)

    The abstract class for Proto builds on ERM

    The main feature of Proto abstract class is to load an additional
    "prototype" model (that is typically pretrained) in addition to the
    featurizer. For each input, the model concatenates the feature and
    prototype, followed by a bottleneck layer and then produces softmaxes.

    HYPERPARAMS:
    ============

    feat_size : dimensionality of prototyper feature (default 2048)
    bottleneck_size : dimensionality of bottleneck layer (default 512)
    data_parallel : use data-parallel processing (torch.Parallel.DataParallel,
        default=True)
    mixup : (only during prototype training) mixup samples from different
        domains, i.e., randomly select two domains and interpolate with random
        weight between 0.2 and 0.8.

    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, use_relu=True):
        super(Proto, self).__init__()

        self.hparams = hparams

        # self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     num_classes,
        #     self.hparams['nonlinear_classifier'])
        # self.network = nn.Sequential(self.featurizer, self.classifier)
        # self.optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams['weight_decay']
        # )

        # initializing constants
        self.nd = num_domains
        self.nc = num_classes

        # initializing architecture parameters
        featurizer = networks.Featurizer(input_shape, self.hparams)
        self.ft_output_size = featurizer.n_outputs
        self.proto_size = int(self.ft_output_size * 0.25)
        self.feat_size = int(self.ft_output_size)

        # initializing hyperparameters
        self.proto_frac = hparams["proto_train_frac"]
        self.epochs = hparams["n_steps"]
        self.proto_epochs = hparams["n_steps_proto"]

        # self.kernel_type = "gaussian"

        # initializing prototyper
        if use_relu:
            self.prototyper = nn.Sequential(
                featurizer,
                nn.ReLU(inplace=False),
                nn.Linear(self.ft_output_size, self.proto_size),
                nn.ReLU(inplace=False),
            )
        else:
            self.prototyper = nn.Sequential(
                featurizer,
                nn.Linear(self.ft_output_size, self.proto_size),
            )

        # # initializing featurizer
        # if use_relu:
        #     self.featurizer = nn.Sequential(
        #         networks.Featurizer(input_shape, self.hparams),
        #         nn.ReLU(inplace=False),
        #         nn.Linear(self.ft_output_size, self.feat_size),
        #         nn.ReLU(inplace=False),
        #     )
        # else:
        #     self.featurizer = nn.Sequential(
        #         networks.Featurizer(input_shape, self.hparams),
        #         nn.Linear(self.ft_output_size, self.feat_size),
        #     )

        # # initializing bottleneck architecture on top of prototyper
        # if use_relu:
        #     self.bottleneck = nn.Sequential(
        #         nn.Linear(
        #             self.feat_size + self.proto_size, self.hparams["bottleneck_size"]
        #         ),
        #         nn.ReLU(inplace=False),
        #     )
        # else:
        #     self.bottleneck = nn.Sequential(
        #         nn.Linear(
        #             self.feat_size + self.proto_size, self.hparams["bottleneck_size"]
        #         )
        #     )

        # # initalizing classifier
        # self.classifier = nn.Linear(self.hparams["bottleneck_size"], num_classes)

        # initialize parameters based on doing prototype training
        # or not

        do_prototype_training = (
            hparams["proto_model"] is None or hparams["train_prototype"]
        )

        # if do_prototype_training:
        params = self.prototyper.parameters()
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.hparams["proto_lr"],
            weight_decay=self.hparams["proto_weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.proto_epochs
        )
        # else:
        #     params = (
        #         list(self.bottleneck.parameters())
        #         + list(self.classifier.parameters())
        #         + list(self.featurizer.parameters())
        #     )
        #     self.optimizer = torch.optim.Adam(
        #         params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        #     )
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.optimizer, self.epochs
        #     )

    def prototype_update(self, minibatches):
        """ Update to train prototypical network. """

        all_x = torch.cat([x for x, y in minibatches])
        all_d = torch.cat(
            [i * torch.ones((y.shape[0])) for i, (x, y) in enumerate(minibatches)]
        )

        x_proto = self.prototyper(all_x)
        num_domains_iter = len(minibatches)
        td = num_domains_iter

        if self.hparams["mixup"] > 0:
            n_mx_dom = int(self.hparams["mixup"] * num_domains_iter)
            n_iter = int(self.hparams["mixup"])

            mx_minibatches = random_pairs_of_minibatches(minibatches)[:n_mx_dom]

            for i in range(n_iter):
                _st = i * num_domains_iter
                _en = (i + 1) * num_domains_iter

                this_batch = None
                for (xi, yi), (xj, _) in mx_minibatches[_st:_en]:

                    alpha = 0.2 + 0.6 * random.random()
                    _x = alpha * xi + (1 - alpha) * xj

                    all_d = torch.cat([all_d, td * torch.ones(yi.shape)])
                    td += 1
                    if this_batch is None:
                        this_batch = _x
                    else:
                        this_batch = torch.cat([this_batch, _x])

                _px = self.prototyper(this_batch)
                x_proto = torch.cat([x_proto, _px])

        loss, accuracy = prototypical_loss(
            x_proto, all_d, int(x_proto.shape[0] / (td * 1.0 / self.proto_frac))
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if torch.is_tensor(loss):
            loss = loss.item()

        if torch.is_tensor(accuracy):
            accuracy = accuracy.item()

        return {"proto_loss": loss, "proto_acc": accuracy}

    def save_prototype(self, output_file):
        """ Write prototype to file. """
        return torch.save(self.prototyper, output_file)

    def load_prototype(self, output_file):
        """ Load prototype from file. """
        self.prototyper = torch.load(output_file)

    # def save_model(self, output_file):
    #     """ Write complete model to file."""
    #     model = [self.prototyper, self.bottleneck, self.featurizer, self.classifier]
    #     return torch.save(model, output_file)

    def compute_average_prototype(self, x):
        """ Compute prototype feature and average it."""
        x = self.prototyper(x)
        return torch.mean(x, dim=0).detach().cpu()

    # def attach_prototypes(self, prototypes):
    #     """ Add prototypes to model. """
    #     self.prototypes = prototypes

    def init_prototype_training(self):
        """ Set up model to train prototype. """

        # self.bottleneck.to("cpu")
        # self.featurizer.to("cpu")
        # self.classifier.to("cpu")

        self.prototyper.to("cuda")
        self.prototyper = nn.parallel.DataParallel(self.prototyper).cuda()

    # def init_main_training(self, hparams):
    #     """ Discard earlier optimizers and prepare for main training. """

    #     # first unload prototyper
    #     self.prototyper.to("cpu")

    #     self.bottleneck.to("cuda")
    #     self.featurizer.to("cuda")
    #     self.classifier.to("cuda")

    #     self.bottleneck = nn.parallel.DataParallel(self.bottleneck)
    #     self.featurizer = nn.parallel.DataParallel(self.featurizer)
    #     self.classifier = nn.parallel.DataParallel(self.classifier)

    #     params = (
    #         list(self.bottleneck.parameters())
    #         + list(self.classifier.parameters())
    #         + list(self.featurizer.parameters())
    #     )

    #     self.optimizer = torch.optim.Adam(
    #         params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
    #     )

    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         self.optimizer, self.epochs
    #     )

    # def predict(self, x, idx, device):
    #     """ Forward function to compute output. """

    #     bs = x.shape[0]
    #     proto_tile = self.prototypes[idx].to(device).unsqueeze(0).repeat(bs, 1)

    #     x = self.featurizer(x)
    #     x = torch.cat([x, proto_tile], dim=1)
    #     x = self.bottleneck(x)
    #     x = self.classifier(x)

    #     return x

    # def update(self, minibatches, device, base=0):
    #     """ Update step to be applied during main training."""
    #     all_x = torch.cat([x for x, y in minibatches])
    #     all_y = torch.cat([y for x, y in minibatches])
    #     nmb = len(minibatches) # 4

    #     all_proto = None

    #     for idx in range(len(minibatches)):
    #         bs = minibatches[idx][0].shape[0]
    #         px = self.prototypes[base + idx].to(device).unsqueeze(0).repeat(bs, 1) # px.shape=[12,512]

    #         if all_proto is None:
    #             all_proto = px
    #         else:
    #             all_proto = torch.cat([all_proto, px], dim=0)

    #     x = self.featurizer(all_x)
    #     bs = x.size(0)/nmb
    #     penalty = 0.0

    #     def cu(y):
    #         return min(y, x.size(0))

    #     # adding mmd loss
    #     if self.hparams["mmd_gamma"] > 0.0:
    #         st_i = 0
    #         for i in range(nmb):
    #             en_i = cu(st_i + minibatches[i][0].size(0))
    #             st_j = cu(en_i)
    #             for j in range(i + 1, nmb):
    #                 en_j = cu(st_j + minibatches[j][0].size(0))
    #                 mmd_ij = self.mmd(x[st_i : en_i], x[st_j : en_j])
    #                 penalty += mmd_ij
    #                 st_j = en_j
    #             st_i = en_i

    #     if nmb > 1:
    #         penalty /= nmb * (nmb - 1) / 2

    #     x = torch.cat([x, all_proto], dim=1)
    #     x = self.bottleneck(x)
    #     x = self.classifier(x)

        
    #     loss = cross_entropy(x, all_y)
    #     if self.hparams["mmd_gamma"] > 0:
    #         loss += self.hparams["mmd_gamma"] * penalty

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.scheduler.step()

    #     return {"loss": loss.item()}
    
    # # adding MMD functionality 
    # def my_cdist(self, x1, x2):
    #     x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    #     x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    #     res = torch.addmm(
    #         x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    #     ).add_(x1_norm)
    #     return res.clamp_min_(1e-30)

    # def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
    #     D = self.my_cdist(x, y)
    #     K = torch.zeros_like(D)

    #     for g in list(gamma):
    #         K.add_(torch.exp(D.mul(-g)))

    #     return K

    # def mmd(self, x, y):
    #     if self.kernel_type == "gaussian":
    #         Kxx = self.gaussian_kernel(x, x).mean()
    #         Kyy = self.gaussian_kernel(y, y).mean()
    #         Kxy = self.gaussian_kernel(x, y).mean()
    #         return Kxx + Kyy - 2 * Kxy
    #     else:
    #         mean_x = x.mean(0, keepdim=True)
    #         mean_y = y.mean(0, keepdim=True)
    #         cent_x = x - mean_x
    #         cent_y = y - mean_y
    #         cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    #         cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    #         mean_diff = (mean_x - mean_y).pow(2).mean()
    #         cova_diff = (cova_x - cova_y).pow(2).mean()

    #         return mean_diff + cova_diff


class Proto_NoReLU(Proto):
    """ Implements Proto without ReLU. """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Proto_NoReLU, self).__init__(
            input_shape, num_classes, num_domains, hparams, use_relu=False
        )


class Proto_Mixup(Proto):
    """ Implements Proto with Mixup=1. """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        hparams["mixup"] = 1
        hparams["batch_size"] = int(hparams["batch_size"] * 0.5)
        super(Proto_Mixup, self).__init__(
            input_shape, num_classes, num_domains, hparams, use_relu=False
        )


class Proto_NoReLU_Mixup(Proto_NoReLU):
    """ Implements Proto without ReLU with Mixup=1. """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        hparams["mixup"] = 1
        hparams["batch_size"] = int(hparams["batch_size"] * 0.5)
        super(Proto_NoReLU_Mixup, self).__init__(
            input_shape, num_classes, num_domains, hparams, use_relu=False
        )