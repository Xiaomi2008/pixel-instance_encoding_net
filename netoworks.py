#import tensorflow as tf
import six
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch_networks import torch_networks as torch_net
from label_transform import volumes
from label_transform.volumes import HDF5Volume
from label_transform.volumes import bounds_generator
from label_transform.volumes import SubvolumeGenerator

model = torch_net.Unet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
data_config = 'conf/cremi_datasets.toml'
volumes = HDF5Volume.from_toml(data_config)
V_1 = volumes[volumes.keys()[0]]
def train():
    model.train()
    bounds_gen=bounds_generator(V1.shape,[10,320,320])
    sub_vol_gen =SubvolumeGenerator(V1,bounds_gen)
    for i in xrange(200):
        C = six.next(sub_vol_gen);
        labels = C['label_dataset']
        images = C['image_dataset']
        data, target = Variable(images), Variable(labels)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
