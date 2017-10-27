import os, sys
sys.path.append('../')
import torch
def l2_norm(x):
    #epsilon=torch.cuda.DoubleTensor([1e-12])
    #sq_x   = torch.max(x**2,epsilon)
    #sq_x   = torch.max(x**2,epsilon)
    #e_mat  = torch.zero_like(sq_x)
    sum_x  = torch.sum(x**2,1,keepdim=True)
    sqrt_x = torch.sqrt(sum_x)
    return x/sqrt_x

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))

def angularLoss(pred, gt, weight=0, outputChannels=2):
    pred        = l2_norm(pred)*0.9999999999
    gt          = l2_norm(gt)*0.9999999999
    prod_sum    = torch.sum(gt*pred,1)
    angle_err   = torch.acos(prod_sum)
    loss        = torch.sum(angle_err*angle_err)
    return loss