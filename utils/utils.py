
import os, sys
sys.path.insert(0,'../')
import numpy as np
import torch
import scipy.sparse as sparse
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import disk, skeletonize
import skimage
from skimage.filters import gaussian
from torch.autograd import Variable
from munkres import Munkres
from evaluation import adapted_rand
from orig_cremi_evaluation import voi
import pdb

def evalute_pred_dist_with_threshold(seg_t,distance, thresholds=None):
    thresholds = np.linspace(0.5,15,10) if thresholds is None else thresholds
    for th in thresholds:
        #d_seg= watershed_seg2(d_orig[100:,:,:], d_combine[100:,:,:], threshold = th)
        d_seg= watershed_seg(distance, threshold = th)
        #pdb.set_trace()
        #d_seg= watershed_seg(d_orig[100:,:,:], threshold = th)
        arand = adapted_rand(d_seg.astype(np.int), seg_t)
        split, merge = voi(d_seg.astype(np.int), seg_t)
        #arands.append(arand)
        print('arand, split, merge = {:.3f}, {:.3f}, {:.3f} \
            for threshold = {:.3f}'.format(arand,split,merge,th))

def watershed_seg(distance,threshold = 2.0):
    if isinstance(distance, Variable):
        distance = distance.data
    if isinstance(distance, torch.FloatTensor):
        distance = distance.cpu().numpy()
    distance = np.squeeze(distance)
    markers = distance > threshold
    markers = skimage.morphology.label(markers)
    seg_labels = watershed(-distance, markers)
    return seg_labels


def watershed_seg2(distance_1,distance_fuse,threshold = 2.0):
    if isinstance(distance_1, Variable):
        distance_1 = distance_1.data
    if isinstance(distance_fuse, Variable):
        distance_fuse = distance_fuse.data
    
    if isinstance(distance_fuse, torch.FloatTensor):
        distance_fuse = distance_fuse.cpu().numpy()
    if isinstance(distance_1, torch.FloatTensor):
        distance_1 = distance_1.cpu().numpy()
    #distance_fuse = distance_fuse.cpu().numpy()
    distance_fuse = np.squeeze(distance_fuse)
    markers = distance_fuse > threshold
    markers = skimage.morphology.label(markers)

    #distance_1 = distance_1.cpu().numpy()
    distance_1 = np.squeeze(distance_1)
    


    seg_labels = watershed(-distance_1, markers)
    return seg_labels


if __name__ == '__main__':
    A =np.ones([22,1250,1250])
    B =np.ones([22,1250,1250])
    B[10:20,:,:]=6
    evalute_pred_dist_with_threshold(A,B)


