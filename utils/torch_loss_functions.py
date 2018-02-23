import os, sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
from torch.autograd import Variable


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
    return torch.mean(boudary * (0 - input) ** 2 + 0.5 * (1 - boudary) * (torch.abs(input - target)))


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
            smooth = 1.0
            iflat = input.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()
            return 1.0 - (((2. * intersection + smooth) /
                   (iflat.sum() + tflat.sum() + smooth)))


class DiceCHLoss(nn.Module):
    def __init__(self):
            super(DiceCHLoss,self).__init__()

    def forward(self, input, target):
            smooth = 1
            #out = torch.sigmoid(input)
            out = input
            iflat = out.view(out.shape[0]*out.shape[1], -1)
            tflat = target.view(target.shape[0]*target.shape[1], -1)
            intersection = (iflat * tflat).sum(1,True)
            return 1.0 - (((2. * intersection + smooth) /
                   (iflat.sum(1,True) + tflat.sum(1,True) + smooth)))

def softIoU(target, out, e=1e-6):

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


    out = torch.sigmoid(out)
    # clamp values to avoid nan loss
    #out = torch.clamp(out,min=e,max=1.0-e)
    #target = torch.clamp(target,min=e,max=1.0-e)

    num = (out*target).sum(1,True)
    den = (out+target-out*target).sum(1,True) + e
    iou = num / den
    # set iou to 0 for masks out of range
    # this way they will never be picked for hungarian matching
    cost = (1 - iou)

    return cost.squeeze()


class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_true, y_pred, sw=None):
        costs = softIoU(y_true,y_pred).view(-1,1)
        costs = torch.mean(costs)
        return costs






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


class softIOU_match_loss(nn.Module):
    def __init__(self):
        super(softIOU_match_loss,self).__init__()
        #self.dice_loss =DiceLoss()
        #self.dice_loss =softIoULoss()
        self.dice_loss =DiceCHLoss()

    def forward(self, input, target):
        t_masks  = self.target_seg_to_mask(target)
        t_masks  = Variable(torch.from_numpy(t_masks).float().cuda())
        #print('start compute oerlap ...')
        overlaps = self.compute_overlaps(input,t_masks)
        #print('done compute overlap ...')
        t_masks, permute_indices = match([t_masks,input],overlaps)
        #print('t_mask_shape = {}'.format(t_masks.shape))
        t_masks_v = Variable(torch.from_numpy(t_masks).float().cuda())
        loss=self.dice_loss(input,t_masks_v)
        #loss = self.dice_loss(t_masks_v.view(-1,t_masks.shape[-1]), input.view(-1,input.shape[-1]))
        loss = torch.mean(loss)

        return loss,t_masks_v.data.cpu()
    
    def target_seg_to_mask(self,target_seg):
        targ_data  = target_seg.data.cpu().numpy()
        targ_shape = targ_data.shape
        mask = np.zeros([targ_shape[0],30,targ_shape[2],targ_shape[3]])
        for b in range(targ_shape[0]):
            t_ids,counts = np.unique(targ_data[b],return_counts = True)
            sort_ids=np.argsort(counts)
            t_ids = t_ids[sort_ids][::-1]
            for i in range(min(30,len(t_ids))):
                mask[b,i] = (targ_data[b] == t_ids[i]).astype(int)
        return mask
    
    def compute_overlaps(self, masks_p, masks_t):
        masks_pdata = masks_p.data
        batch_size = masks_pdata.shape[0]
        #overlap_loss = torch.from_numpy(np.ones([batch_size,300,300])*100)
        overlap_loss = torch.ones(batch_size,30,30)*100

        ch = masks_t.data.shape[1]
        masks_t = masks_t.data
        for i in range(ch):
            #mask_p = masks_p[::,i:i+1,::,::]
            mask_p = masks_pdata[::,i:i+1,::,::]
            mask_p = mask_p.repeat(1, ch, 1, 1)
            #print('mask_t shape = {}'.format(masks_t.shape))
            #print('mask_p shape = {}'.format(mask_p.shape))
            #mask_p = mask_p,view()
            c =  self.dice_loss(mask_p, masks_t)

            # mask_p = mask_p.view(mask_p.size(0)*mask_p.size(1), mask_p.size(2)*mask_p.size(3))
            # mask_t = masks_t.view(masks_t.size(0)*masks_t.size(1), masks_t.size(2)*masks_t.size(3))
            # c= softIoU(mask_p,mask_t)
            c=c.view(-1,ch)
            overlap_loss[:,:,i] = c.cpu()
            #overlap_loss[:,:,i] = c.cpu().data
        return overlap_loss

        # for i in range(batch_size):
        #     mask_pdata = masks_pdata[i]
        #     mask_tdata = masks_t[i]
        #     ch,h,w = mask_tdata.shape
        #     #print('shape = {}'.format(mask_tdata.shape))
        #     #
        #     t_mask =mask_tdata
        #     for c in range(ch):
        #         p_mask = mask_pdata[c]
        #         p_mask = p_mask.repeat(1,ch,1,1)

        #     for j in range(ch): 
        #         t_mask = mask_tdata[j]

        #         for c in range(ch):
        #             p_mask = mask_pdata[c]
        #             overlap_loss[i,j,c] = self.dice_loss(p_mask,t_mask)
                # repeat predicted mask as many times as elements in ground truth.
       
        # to compute iou against all ground truth elements at once
        # y_pred_i = out_mask.unsqueeze(0)
        # y_pred_i = y_pred_i.permute(1,0,2)
        # y_pred_i = y_pred_i.repeat(1,y_mask.size(1),1)
        # y_pred_i = y_pred_i.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))
        # y_true_p = y_mask.view(y_mask.size(0)*y_mask.size(1),y_mask.size(2))

        # c = softIoU(y_true_p, y_pred_i)
        # c = c.view(sw_mask.size(0),-1)
        # scores[:,:,t] = c.cpu().data
        

        # return overlap_loss


    # def compute_overlaps(self, inputs, targets):
    #     """
    #     Args:
    #        input   -[batch,channels,height,width], where each channel is a predicted object mask [0,1]
    #        target  -[batch,ch,height,width], contain integer with each represents a object

    #     """

    #     targ_data  = targets.data.cpu().numpy()
    #     input_data = inputs.data.cpu().numpy()
    #     batch_size =targ_data.shape[0]
    #     # compute objects in the target for every targets in batch
    #     overlaps = np.ones(batch_size,1000,1000)*100
    #     for i in range(len(targ_data)):
    #         targ  = targ_data[i]
    #         t_ids = np.unique(targ)
    #         inpu_d = input_data[i]
    #         ch,h,w = input.shape
    #         for j in range(len(t_ids)):
    #             t_mask = (targ == t_ids[j]).astype(np.int)
    #             for c in range(ch):
    #                 overlap[i,j,c] = self.dice_loss(inpu_d[c],t_mask)
    #         #sort_ids=np.argsort(np.mean(overlap[i,:,:],axis = 1))
    #         #overlap[i,:,:] = overlaps[i,:,sort_idst]





def match(masks, overlaps):
    """
    Code reference from 'heighttps://github.com/Xiaomi2008/rsis/blob/master/src/utils/hungarian.py'
    Args:
        masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
        overlaps - [batch_size,T,T] - matrix of costs between all pairs
    Returns:
        t_mask_cpu - [batch_size,T,N] permuted ground truth masks
        permute_indices - permutation indices used to sort the above
    """

    #overlaps = (overlaps.data).cpu().numpy().tolist()

    overlaps = overlaps.cpu().numpy().tolist()
    m = Munkres()

    t_mask, p_mask = masks

    # get true mask values to cpu as well
    #t_mask_cpu = (t_mask.data).cpu().numpy()
    t_mask_cpu = (t_mask.data).cpu().numpy()
    # init matrix of permutations
    permute_indices = np.zeros((t_mask.size(0), t_mask.size(1)), dtype=int)
    # we will loop over all samples in batch (must apply munkres independently)
    for sample in range(p_mask.size(0)):
        # get the indexes of minimum cost
        indexes = m.compute(overlaps[sample])
        for row, column in indexes:
            # put them in the permutation matrix
            permute_indices[sample, column] = row

        # sort ground according to permutation
        t_mask_cpu[sample] = t_mask_cpu[sample, permute_indices[sample], :]
    
    return t_mask_cpu, permute_indices

