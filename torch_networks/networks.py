import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models
import sys
import math
import pdb
class Downblock(nn.Module):
    def __init__(self,in_ch,num_conv,ch_growth_rate,kernel_size = 3):
        super(Downblock, self).__init__()
        assert(num_conv>0)
        self.in_ch = in_ch
        self.num_conv =num_conv
        self.ch_growth_rate =ch_growth_rate
        self.kernel_size =kernel_size
        self.layers=self.build_layer_block()
        self.block = nn.Sequential(*self.layers)
    def forward(self,x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2
        
        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch * self.ch_growth_rate
            else:
                self.in_ch = out_ch
            # conv = nn.Conv2d(self.in_ch, out_ch,self.kernel_size=kernel_size, padding=same_padding)
            # print self.in_ch, out_ch
            # conv = nn.Conv2d(self.in_ch, out_ch)
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False))
        return layers


class Upblock(nn.Module):
    def __init__(self, in_ch, num_conv,ch_down_rate,kernel_size = 3):
        super(Upblock, self).__init__()
        assert(num_conv>0)
        self.in_ch = in_ch
        self.num_conv =num_conv
        self.ch_down_rate =ch_down_rate
        self.kernel_size =kernel_size
        self.layers=self.build_layer_block()
        self.block = nn.Sequential(*self.layers)

        

    def forward(self,x):
        return self.block(x)

    def build_layer_block(self):
        layers = []
        same_padding = self.kernel_size // 2
        print (self.num_conv)
        for i in range(self.num_conv):
            if i == 0:
                out_ch = self.in_ch // self.ch_down_rate
            else:
                self.in_ch = out_ch
            print(self.in_ch,self.kernel_size,same_padding)
            layers.append(nn.Conv2d(self.in_ch, out_ch, kernel_size=self.kernel_size, padding=same_padding))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        return layers

class Unet(nn.Module):
    def __init__(self, in_ch =1, first_out_ch=16, out_ch =1, number_bolck=4,num_conv_in_block=2,ch_change_rate=2,kernel_size = 3):
        super(Unet, self).__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        #self.ch_down_rate =ch_down_rate
        
        self.conv_2d_1 = nn.Conv2d(in_ch, first_out_ch, kernel_size=kernel_size, padding=1)
        
        self.down_block_1 = Downblock(first_out_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b2_down_ch = first_out_ch * ch_change_rate
        self.down_block_2 = Downblock(b2_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b3_down_ch= b2_down_ch * ch_change_rate
        self.down_block_3 = Downblock(b3_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        
        b4_down_ch = b3_down_ch * ch_change_rate
        self.down_block_4 = Downblock(b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)
        

        b1_up_ch = b4_down_ch * ch_change_rate
        self.up_block_1 = Upblock(b1_up_ch+b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b2_up_ch = b1_up_ch // ch_change_rate
        self.up_block_2 = Upblock(b2_up_ch+b4_down_ch,num_conv_in_block,ch_change_rate,kernel_size)

        b3_up_ch = b2_up_ch // ch_change_rate
        self.up_block_3 = Upblock(160,num_conv_in_block,ch_change_rate,kernel_size)

        b4_up_ch = b3_up_ch // ch_change_rate
        self.up_block_4 = Upblock(96,num_conv_in_block,ch_change_rate,kernel_size)

        last_up_ch = b4_up_ch // ch_change_rate
        self.finnal_conv2d = nn.Conv2d(last_up_ch, out_ch, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.finnal_conv2d = nn.Conv2d(48, 2, kernel_size=3, padding=1)

    def forward(self,x):
        #x=self.finnal_conv2d(x)
        x1  = self.conv_2d_1(x)
        d_1 = self.down_block_1(x1)
        d_2 = self.down_block_2(d_1)
        d_3 = self.down_block_3(d_2)
        d_4 = self.down_block_4(d_3)
        c_1 = torch.cat((self.upsample(d_4), d_3), 1)
        u_1 = self.up_block_1(c_1)

        c_2 = torch.cat((self.upsample(u_1), d_2), 1)
        u_2 = self.up_block_2(c_2)

        c_3 = torch.cat((self.upsample(u_2), d_1), 1)
        u_3 = self.up_block_3(c_3)

        c_4 = torch.cat((self.upsample(u_3), x1), 1)
        u_4 = self.up_block_4(c_4)

        out = self.finnal_conv2d(u_4)
        return out




def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))

def l2_norm(x):
    #return x/torch.sqrt(torch.max(x**2,0)[0])
    return x/torch.sqrt(torch.max(x**2,0)[0])
def angularLoss(pred, gt, weight=0, outputChannels=2):

    # pred =  

    pred        = pred.transpose(1,2).transpose(2,3).contiguous()
    gt          = gt.transpose(1,2).transpose(2,3).contiguous()
    pred        = pred.view(-1, outputChannels)
    gt          = gt.view(-1, outputChannels)
    #weight      = weight.view(-1, 1).float()
    pred        = l2_norm(pred)*0.999999
    gt          = l2_norm(gt)*0.999999
    p_xy  =pred[:,0]/torch.sqrt(torch.sum((pred*pred),1))   
    gt_xy =gt[:,0]/torch.sqrt(torch.sum((gt*gt),1))
    err_angle= torch.acos(p_xy) - torch.acos(gt_xy)
    loss = torch.sum(err_angle*err_angle)
    return loss

def test_angularLoss():
    A = torch.randn(10,10,2).cuda
    B = A
    W = torch.randn(10,10,1)
    W = torch.abs(l2_norm(W))
    print(angularLoss(A,B,W))



    # errorAngles =torch.acos(pred*gt)
    # errorAngles = tf.acos(tf.reduce_sum(pred * gt, reduction_indices=[1], keep_dims=True))

    # lossAngleTotal = tf.reduce_sum((tf.abs(errorAngles*errorAngles))*weight)
    # return lossAngleTotal





# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(functional.log_softmax(inputs), targets)

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


if __name__ == '__main__':
    test_angularLoss()
    # net = Unet()
    # print(net)

    # test_x = Variable(torch.FloatTensor(1, 1, 1024, 1024))
    # out_x = net(test_x)

    # print(out_x.size())
