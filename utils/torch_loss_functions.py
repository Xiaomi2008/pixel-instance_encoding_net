import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
def l2_norm(x):
    epsilon = 1e-12
    #epsilon=torch.cuda.DoubleTensor([1e-12])
    #sq_x   = torch.max(x**2,epsilon)
    #sq_x   = torch.max(x**2,epsilon)
    #e_mat  = torch.zero_like(sq_x)
    sum_x  = torch.sum(x**2,1,keepdim=True)
    sqrt_x = torch.sqrt(sum_x).clamp(min=epsilon).expand_as(x)
    return x/sqrt_x

def dice_loss(input, target):
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))

def angularLoss(pred, gt, weight=0, outputChannels=2):
   
   # pred        = l2_norm(pred)*0.9999999999
   # gt          = l2_norm(gt)*0.9999999999

    pred        = F.normalize(pred)*0.999999
    gt          = F.normalize(gt)*0.999999

    prod_sum    = torch.sum(gt*pred,1)
    angle_err   = torch.acos(prod_sum)
    loss        = torch.sum(angle_err*angle_err)
    return loss

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
    #loss = F.nll_loss(F.log_softmax(input), target, weight=weight, size_average=False)
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= target_mask.sum().data[0]

    return loss