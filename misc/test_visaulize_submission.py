import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import h5py as h5
def show_figure(seg3D,raw):
        my_dpi = 96
        fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
        label2d_seg = label2rgb(seg3D)    
        a = fig.add_subplot(2, 2, 1)
        plt.imshow(label2d_seg[0], interpolation='nearest')
        a.set_title('upper_seg')    
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(label2d_seg[1], interpolation='nearest')
        a.set_title('lower_seg')

        a = fig.add_subplot(2, 2, 3)
        plt.imshow(raw[0], interpolation='nearest')
        a.set_title('upper_raw')    
        a = fig.add_subplot(2, 2, 4)
        plt.imshow(raw[1], interpolation='nearest')
        a.set_title('lower_raw')
        plt.show()

seg_file  =  '../submission/back/Set_A.hdf'
raw_file =  '../data/sample_A+_20160601.hdf'
h5f_seg = h5.File(seg_file)
segs = np.array(h5f_seg['volumes/labels/neuron_ids'])

h5f_raw = h5.File(raw_file)
raws = np.array(h5f_raw['volumes/raw'])

show_figure(segs,raws)
