import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from IPython import embed


class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class CenterWeightTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterWeightTripletLoss, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.tv_loss = nn.MarginRankingLoss(margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        N = dist.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist * is_pos
        dist_an = dist * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)
        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class CenterWeightTripletLoss1(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterWeightTripletLoss1, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        targets_ = labels.chunk(3, 0)
        feats = feats.chunk(3, 0)

        centers = []
        label_uni = labels.unique()
        for i in range(len(feats)):
            feat_i = feats[i]
            for lab in label_uni:
                idx = torch.where(targets_[i] == lab)[0]
                feat = feat_i[idx]
                centers.append(torch.mean(feat, dim=0, keepdim=True))

        inputs = torch.cat(centers)
        targets = torch.cat([label_uni, label_uni, label_uni])

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        N = dist.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist * is_pos
        dist_an = dist * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)
        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class CenterTVTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterTVTripletLoss, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.tv_loss = nn.MarginRankingLoss(margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        label_num = len(label_uni)

        n = feats.shape[0] // 3
        rgb_feat = feats.narrow(0, 0, n)
        ca_feat = feats.narrow(0, n, n)
        ir_feat = feats.narrow(0, 2 * n, n)
        rgb_feat_chunk = rgb_feat.chunk(label_num, 0)
        ca_feat_chunk = ca_feat.chunk(label_num, 0)
        ir_feat_chunk = ir_feat.chunk(label_num, 0)
        rgb_center = []
        ca_center = []
        ir_center = []
        for i in range(label_num):
            rgb_center.append(torch.mean(rgb_feat_chunk[i], dim=0, keepdim=True))
            ca_center.append(torch.mean(ca_feat_chunk[i], dim=0, keepdim=True))
            ir_center.append(torch.mean(ir_feat_chunk[i], dim=0, keepdim=True))
        rgb_center = torch.cat(rgb_center).unsqueeze(0)
        ca_center = torch.cat(ca_center).unsqueeze(0)
        ir_center = torch.cat(ir_center).unsqueeze(0)
        centers = torch.concat([rgb_center, ca_center, ir_center], dim=0)

        [n, m, d] = centers.shape
        tv_h = torch.sum(torch.abs(centers[1:, :, :] - centers[:-1, :, :])) / (m * n * d)
        tv_v = torch.sum(torch.abs(centers[:, 1:, :] - centers[:, :-1, :])) / (m * n * d)
        tv_loss = torch.exp(tv_h) + torch.exp(-tv_v)
        return tv_loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class CenterClusterCompactLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(CenterClusterCompactLoss, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        intra_center = []
        intra_dist = []
        for i in range(label_num * 3):
            samples = feat[i]
            center = torch.mean(samples, dim=0, keepdim=True)
            dist_i = euclidean_dist(samples, center)
            max_v = torch.max(dist_i, dim=0, keepdim=True)[0]
            diff = dist_i - max_v
            intra_dist.append(torch.mean(torch.exp(diff), dim=0, keepdim=True))
            intra_center.append(center)
        intra_dist = torch.cat(intra_dist)
        intra_loss = torch.mean(intra_dist)

        intra_center = torch.cat(intra_center)
        centers = intra_center.chunk(3, 0)
        center_vt = (centers[0] + centers[1]) / 2.0
        center_it = (centers[2] + centers[1]) / 2.0
        inter_center = (center_vt + center_it) / 2.0

        inter_dist = euclidean_dist(inter_center, intra_center)

        M, N = inter_dist.size()
        # shape [N, N]
        is_pos_temp = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg_temp = targets.expand(N, N).ne(targets.expand(N, N).t()).float()
        is_pos = is_pos_temp[:M, :]
        is_neg = is_neg_temp[:M, :]

        dist_ap = inter_dist * is_pos
        dist_an = inter_dist * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        inter_loss = self.ranking_loss(closest_negative - furthest_positive, y)
        # compute accuracy
        # correct = torch.ge(closest_negative, furthest_positive).sum().item()

        c3_loss = intra_loss + inter_loss
        return c3_loss

class CenterDCLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(CenterDCLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss1 = nn.SoftMarginLoss()

    def forward(self, out, labels):
        n = out.shape[0] // 3
        out1 = out.narrow(0, 0, n)      # rgb
        out2 = out.narrow(0, n, n)      # ca
        out3 = out.narrow(0, 2 * n, n)  # ir

        label = labels.narrow(0, 0, n)

        is_pos = label.expand(n, n).eq(label.expand(n, n).t()).float()
        is_neg = label.expand(n, n).ne(label.expand(n, n).t()).float()

        out1_pos_stds, out1_neg_stds = [], []
        out2_pos_stds, out2_neg_stds = [], []
        out3_pos_stds, out3_neg_stds = [], []
        for i in range(n):
            out1_pos_samples = out1[torch.where(is_pos[i] == 1)]
            out1_neg_samples = out1[torch.where(is_neg[i] == 1)]
            out1_pos_mu = out1_pos_samples.mean(0)
            out1_neg_mu = out1_neg_samples.mean(0)
            out1_pos_std = ((out1[i] - out1_pos_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
            out1_neg_std = ((out1[i] - out1_neg_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
            out1_pos_stds.append(out1_pos_std)
            out1_neg_stds.append(out1_neg_std)

            out2_pos_samples = out2[torch.where(is_pos[i] == 1)]
            out2_neg_samples = out2[torch.where(is_neg[i] == 1)]
            out2_pos_mu = out2_pos_samples.mean(0)
            out2_neg_mu = out2_neg_samples.mean(0)
            out2_pos_std = ((out2[i] - out2_pos_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
            out2_neg_std = ((out2[i] - out2_neg_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
            out2_pos_stds.append(out2_pos_std)
            out2_neg_stds.append(out2_neg_std)

            out3_pos_samples = out3[torch.where(is_pos[i] == 1)]
            out3_neg_samples = out3[torch.where(is_neg[i] == 1)]
            out3_pos_mu = out3_pos_samples.mean(0)
            out3_neg_mu = out3_neg_samples.mean(0)
            out3_pos_std = ((out3[i] - out3_pos_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
            out3_neg_std = ((out3[i] - out3_neg_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
            out3_pos_stds.append(out3_pos_std)
            out3_neg_stds.append(out3_neg_std)

        out1_pos_stds = torch.cat(out1_pos_stds)
        out1_neg_stds = torch.cat(out1_neg_stds)
        out2_pos_stds = torch.cat(out2_pos_stds)
        out2_neg_stds = torch.cat(out2_neg_stds)
        out3_pos_stds = torch.cat(out3_pos_stds)
        out3_neg_stds = torch.cat(out3_neg_stds)

        y = out1_pos_stds.new().resize_as_(out1_pos_stds).fill_(1)
        cdc_std_loss = self.ranking_loss(out1_neg_stds, out1_pos_stds, y) \
                   + self.ranking_loss(out2_neg_stds, out2_pos_stds, y) \
                   + self.ranking_loss(out3_neg_stds, out3_pos_stds, y)

        label_uni = label.unique()
        label_num = len(label_uni)
        out1_ = out1.chunk(label_num, 0)
        out2_ = out2.chunk(label_num, 0)
        out3_ = out3.chunk(label_num, 0)
        center_out1 = []
        center_out2 = []
        center_out3 = []
        for i in range(label_num):
            out1_i = F.softmax(out1_[i], dim=1)
            out2_i = F.softmax(out2_[i], dim=1)
            out3_i = F.softmax(out3_[i], dim=1)
            center_out1.append(torch.mean(out1_i, dim=0, keepdim=True))
            center_out2.append(torch.mean(out2_i, dim=0, keepdim=True))
            center_out3.append(torch.mean(out3_i, dim=0, keepdim=True))

        center_out1 = torch.cat(center_out1)
        center_out2 = torch.cat(center_out2)
        center_out3 = torch.cat(center_out3)

        P_1 = (center_out1 + center_out2) / 2.0
        P_2 = (center_out3 + center_out2) / 2.0

        cdc_js_loss = 0.5 * ((center_out1 * (center_out1.log() - P_1.log())).sum(1).sum() / center_out1.size()[0]) + 0.5 * ((center_out2 * (center_out2.log() - P_1.log())).sum(1).sum() / center_out2.size()[0]) + \
                      0.5 * ((center_out3 * (center_out3.log() - P_2.log())).sum(1).sum() / center_out3.size()[0]) + 0.5 * ((center_out2 * (center_out2.log() - P_2.log())).sum(1).sum() / center_out2.size()[0])
        cdc_loss = cdc_std_loss + cdc_js_loss
        return cdc_loss


class DivLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(DivLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        # self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss1 = nn.SoftMarginLoss()

    def forward(self, feats, labels):
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        center = torch.cat(center)

        N = label_num * 3
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        pos_stds = []
        neg_stds = []
        for i in range(len(center)):
            pos_center = center[torch.where(is_pos[i] == 1)]
            neg_center = center[torch.where(is_neg[i] == 1)]

            pos_mu = pos_center.mean(0)
            pos_std = ((pos_center - pos_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()

            neg_mu = neg_center.mean(0)
            neg_std = ((pos_center - neg_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()

            pos_stds.append(pos_std)
            neg_stds.append(neg_std)

        # pos_stds = torch.cat(pos_stds)
        # neg_stds = torch.cat(neg_stds)
        # stds_gap = (neg_stds - pos_stds).mean(dim=1)
        # y = stds_gap.new().resize_as_(stds_gap).fill_(1)
        # pos_gap = pos_stds.mean(dim=1)
        # std_loss = self.ranking_loss(stds_gap, y) + pos_gap.mean(0)

        pos_stds = torch.cat(pos_stds)
        neg_stds = torch.cat(neg_stds)
        pos_stds_mean = pos_stds.mean(1)
        neg_stds_mean = neg_stds.mean(1)
        y = pos_stds_mean.new().resize_as_(pos_stds_mean).fill_(1)
        # y1 = pos_stds_mean.new().resize_as_(pos_stds_mean).fill_(-1)
        std_loss = self.ranking_loss(neg_stds_mean, pos_stds_mean, y)
        return std_loss

class DivLoss1(nn.Module):
    def __init__(self, margin=0.3):
        super(DivLoss, self).__init__()
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss1 = nn.SoftMarginLoss()

    def forward(self, feats, labels):
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        center = torch.cat(center)

        N = label_num * 3
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        pos_stds = []
        neg_stds = []
        for i in range(len(center)):
            pos_center = center[torch.where(is_pos[i] == 1)]
            neg_center = center[torch.where(is_neg[i] == 1)]

            pos_mu = pos_center.mean(0)
            pos_std = ((pos_center - pos_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()

            neg_mu = neg_center.mean(0)
            neg_std = ((neg_center - neg_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()

            pos_stds.append(pos_std)
            neg_stds.append(neg_std)

        pos_stds = torch.cat(pos_stds)
        neg_stds = torch.cat(neg_stds)
        stds_gap = (neg_stds - pos_stds).mean(dim=1)
        y = stds_gap.new().resize_as_(stds_gap).fill_(1)
        # pos_gap = pos_stds.mean(dim=1)
        # y1 = stds_gap.new().resize_as_(stds_gap).fill_(-1)
        std_loss = self.ranking_loss(stds_gap, y)

        # pos_stds = torch.cat(pos_stds)
        # neg_stds = torch.cat(neg_stds)
        # pos_stds_mean = pos_stds.mean(1)
        # neg_stds_mean = neg_stds.mean(1)
        # y = pos_stds_mean.new().resize_as_(pos_stds_mean).fill_(1)
        # y1 = pos_stds_mean.new().resize_as_(pos_stds_mean).fill_(-1)
        # pos_gap = pos_stds.mean(dim=1)
        # std_loss = self.ranking_loss(neg_stds_mean, pos_stds_mean, y)
        return std_loss

class JSDivLoss(nn.Module):
    def __init__(self):
        super(JSDivLoss, self).__init__()
    def forward(self, out, labels):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number

        n = out.shape[0] // 3
        out1 = out.narrow(0, 0, n)     # rgb
        out2 = out.narrow(0, n, n)     # ca
        out3 = out.narrow(0, 2 * n, n) # ir

        label = labels.narrow(0, 0, n)

        label_uni = label.unique()
        label_num = len(label_uni)
        out1_ = out1.chunk(label_num, 0)
        out2_ = out2.chunk(label_num, 0)
        out3_ = out3.chunk(label_num, 0)
        center_out1 = []
        center_out2 = []
        center_out3 = []
        for i in range(label_num):
            center_out1.append(torch.mean(out1_[i], dim=0, keepdim=True))
            center_out2.append(torch.mean(out2_[i], dim=0, keepdim=True))
            center_out3.append(torch.mean(out3_[i], dim=0, keepdim=True))
        center_out1 = torch.cat(center_out1)
        center_out2 = torch.cat(center_out2)
        center_out3 = torch.cat(center_out3)

        # center_out1 = out1
        # center_out2 = out2
        # center_out3 = out3

        center_out1 = F.softmax(center_out1, dim=1)
        center_out2 = F.softmax(center_out2, dim=1)
        center_out3 = F.softmax(center_out3, dim=1)

        P_1 = (center_out1 + center_out2) / 2.0
        P_2 = (center_out3 + center_out2) / 2.0

        loss = 0.5*((center_out1 * (center_out1.log() - P_1.log())).sum(1).sum() / center_out1.size()[0]) + 0.5*((center_out2 * (center_out2.log() - P_1.log())).sum(1).sum() / center_out2.size()[0]) + \
               0.5*((center_out3 * (center_out3.log() - P_2.log())).sum(1).sum() / center_out3.size()[0]) + 0.5*((center_out2 * (center_out2.log() - P_2.log())).sum(1).sum() / center_out2.size()[0])
        return loss

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    

def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist