"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional as F

MSE = torch.nn.MSELoss()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.03, contrast_mode='all',
                 base_temperature=0.03):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print("exp_logits:", torch.isnan(exp_logits).any())
        inter1= exp_logits.sum(1, keepdim=True)
        # print("inter1:", torch.isnan(inter1).any())
        inter2 = torch.log(inter1)
        # print("inter2:", torch.isnan(inter2).any())
        log_prob = logits - inter2
        # print("log_prob:", torch.isnan(log_prob).any())
        # compute mean of log-likelihood over positive
        inter3 = mask.sum(1)
        # print("inter3:", torch.isnan(inter3).any())
        inter4= (mask * log_prob).sum(1)
        # print(inter4)
        # print("inter4:", torch.isnan(inter4).any())
        mean_log_prob_pos = inter4 / inter3
        # print("mean_log:",torch.isnan(mean_log_prob_pos).any())
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print("loss_1:",torch.isnan(loss).any())
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )
def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.6,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")
def compute_contra_loss(proj1,proj2,pred1,pred2,targets,epoch):
    ri_targets = targets.repeat_interleave(targets.size(0), dim=0)
    re_targets = targets.unsqueeze(1).repeat(targets.size(0), 1).squeeze(1)
    mask = ri_targets.eq(re_targets)
    ex_proj1 = proj1.repeat_interleave(proj1.size(0), dim=0)[mask]
    ex_proj2 = proj2.repeat_interleave(proj2.size(0), dim=0)[mask]
    ex_pred1 = pred1.repeat(pred1.size(0), 1)[mask]
    ex_pred2 = pred2.repeat(pred2.size(0), 1)[mask]
    loss = (batchcosineloss(ex_pred1, ex_proj2) + batchcosineloss(ex_pred2, ex_proj1))/2 +1
    # loss = ((value_mask * (-cosineloss(ex_pred1, ex_proj2) + 2)).mean() + (
    #             value_mask * (-cosineloss(ex_pred2, ex_proj1) + 2)).mean()) / 2
    return loss
def compute_contrava_loss(proj1,proj2,pred1,pred2,targets,epoch):
    ri_targets = targets.repeat_interleave(targets.size(0), dim=0)
    re_targets = targets.unsqueeze(1).repeat(targets.size(0), 1).squeeze(1)
    mask = ri_targets.eq(re_targets)
    ex_proj1 = proj1.repeat_interleave(proj1.size(0), dim=0)[mask]
    ex_proj2 = proj2.repeat_interleave(proj2.size(0), dim=0)[mask]
    ex_pred1 = pred1.repeat(pred1.size(0), 1)[mask]
    ex_pred2 = pred2.repeat(pred2.size(0), 1)[mask]
    # loss = (batchcosineloss(ex_pred1, ex_proj2) + batchcosineloss(ex_pred2, ex_proj1))/2 +1
    loss = (epoch / 10 * batchcosineloss(ex_pred1, ex_proj2) + epoch / 10 * batchcosineloss(ex_pred2, ex_proj1)) / 2 + 1
    # loss = ((value_mask * (-cosineloss(ex_pred1, ex_proj2) + 2)).mean() + (
    #             value_mask * (-cosineloss(ex_pred2, ex_proj1) + 2)).mean()) / 2
    return loss
def compute_adverse_loss(proj1,proj2,pred1,pred2,stu_pred1,stu_pred2,targets,Stu_Matrix,epoch,threshold):
    ri_targets = targets.repeat_interleave(targets.size(0), dim=0)
    re_targets = targets.unsqueeze(1).repeat(targets.size(0), 1).squeeze(1)
    mask = ri_targets.eq(re_targets)
    ex_proj1 = proj1.repeat_interleave(proj1.size(0), dim=0)[mask]
    ex_proj2 = proj2.repeat_interleave(proj2.size(0), dim=0)[mask]
    ex_pred1 = pred1.repeat(pred1.size(0), 1)[mask]
    ex_pred2 = pred2.repeat(pred2.size(0), 1)[mask]
    Tec_Matrix = compute_relation(pred1.detach(),targets)
    VauleMask = compute_valuable_relation(Tec_Matrix,Stu_Matrix,epoch,threshold)
    # disti_loss = torch.mean(VauleMask*(stu_pred1-pred1.detach())^2)+torch.mean(VauleMask*(stu_pred2-pred2.detach())^2)
    # loss = ((VauleMask*(-cosineloss(ex_pred1, ex_proj2)+2)).mean() + (VauleMask*(-cosineloss(ex_pred2, ex_proj1)+2)).mean())/2+disti_loss/2
    loss = ((VauleMask*(-cosineloss(ex_pred1, ex_proj2)+2)).mean() + (VauleMask*(-cosineloss(ex_pred2, ex_proj1)+2)).mean())/2
    return VauleMask,loss
def compute_relation(fea1, targets):
    ri_targets = targets.repeat_interleave(targets.size(0), dim=0)
    re_targets = targets.unsqueeze(1).repeat(targets.size(0), 1).squeeze(1)
    mask = ri_targets.eq(re_targets)
    ex_fea1 = fea1.repeat_interleave(fea1.size(0), dim=0)[mask]
    ex_fea2 = fea1.repeat(fea1.size(0), 1)[mask]
    ReMatrix = cosineloss(ex_fea1, ex_fea2)

    return ReMatrix

def compute_valuable_relation(Tec, Stu,epoch,threshold):
    # difference =torch.abs(Tec - Stu)
    # # difference = Tec - Stu
    # sorted,index = torch.sort(difference,dim=0,descending = True)
    # point = round(len(difference)/3)
    # split_value = sorted[point]
    # mask = torch.gt(difference,split_value)
    # matrix = mask*difference+1
    difference = torch.abs(Tec - Stu)
    sorted, index = torch.sort(difference, dim=0, descending=True)
    point = round(len(difference) / 3)
    split_value = sorted[point]
    mask = torch.gt(difference, split_value)
    stu_sorted, stu_index = torch.sort(Stu, dim=0, descending=False)
    stu_point = round(len(Stu) / 3)
    stu_value = stu_sorted[stu_point]
    stu_mask = torch.lt(Stu, stu_value)
    int_stu_mask = stu_mask.int()
    stu_matrix = 0.5*(1-Stu)*int_stu_mask+1
    # final_mask = stu_mask | mask
    # int_mask = final_mask.int()+stu_matrix
    # init_lr * 0.5 * (1. + math.cos(math.pi * epoch / 300))
    int_mask = mask.int()

    matrix = int_mask* difference
    total = matrix.sum()
    mean = total/point
    result = (matrix/mean+1)* 1 if epoch<=threshold else (matrix/mean+1)*epoch/threshold
    return result


def batchcosineloss(x,y):
    criterion=nn.CosineSimilarity(dim=1).to('cuda')
    return -criterion(x,y).mean()

def sumcosineloss(x,y):
    criterion=nn.CosineSimilarity(dim=1).to('cuda')
    return (-criterion(x, y) + 1)
def cosineloss(x,y):
    criterion=nn.CosineSimilarity(dim=1).to('cuda')
    return criterion(x,y)+1


def compute_angle_loss(oldpred1, oldpred2, pred1, pred2, targets):
    ex_opred1_1 = oldpred1.repeat_interleave(oldpred1.size(0), dim=0)
    ex_opred1_2 = oldpred1.repeat(oldpred1.size(0), 1)
    ex_opred2_1 = oldpred2.repeat_interleave(oldpred2.size(0), dim=0)
    ex_opred2_2 = oldpred2.repeat(oldpred2.size(0), 1)

    ex_pred1_1 = pred1.repeat_interleave(pred1.size(0), dim=0)
    ex_pred1_2 = pred1.repeat(pred1.size(0), 1)
    ex_pred2_1 = pred2.repeat_interleave(pred2.size(0), dim=0)
    ex_pred2_2 = pred2.repeat(pred2.size(0), 1)
    shape = oldpred2.shape[0]

    oldgangel1 = cosineloss(ex_opred1_1, ex_opred1_2).view(shape, shape)
    newangle1 = cosineloss(ex_pred1_1, ex_pred1_2).view(shape, shape)

    oldgangel2 =cosineloss(ex_opred2_1, ex_opred2_2).view(shape, shape)
    newangle2 = cosineloss(ex_pred2_1, ex_pred2_2).view(shape, shape)
    loss = (sumcosineloss(oldgangel1.detach(), newangle1).mean(0) + sumcosineloss(oldgangel2.detach(),
                                                                                            newangle2).mean(0))
    
    # loss = MSE(oldgangel1.detach(), newangle1)+ MSE(oldgangel2.detach(),newangle2)
    return loss

