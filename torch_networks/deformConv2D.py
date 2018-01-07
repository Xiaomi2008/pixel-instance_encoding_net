import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
#from torchvision import models
class Conv2dDeformable(nn.Module):
    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 * regular_filter.in_channels, kernel_size=3,
                                       padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = Variable(grid_w,requires_grad=False)
            self.grid_h = Variable(grid_h,requires_grad=False)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x

class conv3dDeformable(nn.Module):
    def __init__(self, regular_filter, cuda=True):
        super(Conv3dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv3d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv3d(regular_filter.in_channels, 3 * regular_filter.in_channels, kernel_size=3,
                                       padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.grid_d = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h, offset_d = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w, d)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))  # (b*c, h, w, d)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))  # (b*c, h, w, d)
        offset_d = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4]))  # (b*c, h, w, d)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h, grid_d = np.meshgrid(np.linspace(-1, 1, x_shape[4]), np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            grid_d = torch.Tensor(grid_d)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
                grid_d = grid_d.cuda()
            self.grid_w = Variable(grid_w,requires_grad=False)
            self.grid_h = Variable(grid_h,requires_grad=False)
            self.grid_d = Variable(grid_d,requires_grad=False)
        offset_w = offset_w + self.grid_w  # (b*c, h, w, d)
        offset_h = offset_h + self.grid_h  # (b*c, h, w, d)
        offset_d = offset_d + self.grid_d  # (b*c, h, w, d)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), int(x_shape[4])).unsqueeze(1)  # (b*c, 1, h, w,d)
        # not work here, as it only support 4D grid tensor by now
        x = F.grid_sample(x, torch.stack((offset_h, offset_w, offset_d), 4))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]),int(x_shape[4]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x
