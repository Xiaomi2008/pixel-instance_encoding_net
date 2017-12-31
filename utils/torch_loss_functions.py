import os, sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x):
    epsilon = 1e-12
    # epsilon=torch.cuda.DoubleTensor([1e-12])
    # sq_x   = torch.max(x**2,epsilon)
    # sq_x   = torch.max(x**2,epsilon)
    # e_mat  = torch.zero_like(sq_x)
    sum_x = torch.sum(x ** 2, 1, keepdim=True)
    sqrt_x = torch.sqrt(sum_x).clamp(min=epsilon).expand_as(x)
    return x / sqrt_x


# def dice_loss(input, target):
#     smooth = 1.0

#     iflat = input.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()

#     return 1.0 - (((2. * intersection + smooth) /
#                    (iflat.sum() + tflat.sum() + smooth)))


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def boundary_sensitive_loss(input, target, boudary):
    ''' boudary exist when == 1, else  0 for none-boundary '''
    ''' we want the boudary to be more important (more loss) then none-dounary area'''
    return torch.sum(boudary * (0 - input) ** 2 + 0.5 * (1 - boudary) * (torch.abs(input - target)))


def angularLoss(pred, gt, weight=0, outputChannels=2):
    # pred        = l2_norm(pred)*0.9999999999
    # gt          = l2_norm(gt)*0.9999999999

    pred = F.normalize(pred) * 0.99999
    gt = F.normalize(gt) * 0.99999

    prod_sum = torch.sum(gt * pred, 1)
    angle_err = torch.acos(prod_sum)
    loss = torch.sum(angle_err * angle_err)
    return loss
class DiceLoss(nn.Module):
    def __init__(self):
            super(DiceLoss,self).__init__()

    def forward(self, input, target):
            smooth = 5.0
            iflat = input.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()
            return 1.0 - (((2. * intersection + smooth) /
                   (iflat.sum() + tflat.sum() + smooth)))

class StableBCELoss(nn.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()


def StableBalancedMaskedBCE(out, target, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses.squeeze().mean()


# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


# this may be unstable sometimes.Notice set the size_average
def CrossEntropy2d(input, target, weight=None, size_average=False):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]
    # loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss
