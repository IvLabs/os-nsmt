#!/usr/bin/env python3
# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    """
    Loss class deriving from Module for the prototypical loss
    function defined below
    """

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    """
    Inspired by https://github.com/jakesnell/prototypical-networks/
    blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    """
    target_cpu = target.to("cpu")
    input_cpu = input.to("cpu")

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query, selected_classes = [], []
    for cx in classes:
        _nq_x = target_cpu.eq(cx.item()).sum().item() - n_support
        if _nq_x > 0:
            n_query.append(_nq_x)
            selected_classes.append(cx.item())
    n_query = min(n_query)

    classes = [cx for cx in classes if cx.item() in selected_classes]
    n_classes = len(classes)
    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np

    query_stx = list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))
    query_stx = [x[:n_query] for x in query_stx]
    query_idxs = torch.stack(query_stx).view(-1)

    query_samples = input.to("cpu")[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val.unsqueeze(0)


class MMDLoss(Module):
    """
    Loss class deriving from Module for the MMD/Coral loss
    function defined below
    """

    def __init__(self, gamma=1.0, n_domains_batch=4, kernel_type='gaussian'):
        super(MMDLoss, self).__init__()
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.n_domains_batch = n_domains_batch

    def forward(self, input):
        return mmd_loss(input, self.n_domains_batch)
    
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in list(gamma):
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
    
    def mmd_loss(x, nmb):
        
        bs = x.size(0)/nmb
        penalty = 0.0

        def cu(y):
            return min(y, x.size(0))

        st_i = 0
        for i in range(nmb):
            en_i = cu(st_i + bs)
            st_j = cu(en_i)
            for j in range(i + 1, nmb):
                en_j = cu(st_j + bs)
                mmd_ij = self.mmd(x[st_i : en_i], x[st_j : en_j])
                penalty += mmd_ij
                st_j = en_j
            st_i = en_i

        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2
        
        return penalty * self.gamma
