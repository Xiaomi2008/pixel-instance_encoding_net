
import os, sys
sys.path.insert(0,'../')
import numpy as np
import torch
import scipy.sparse as sparse
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, random_walker
from skimage.color import label2rgb
from skimage.morphology import disk, skeletonize
import skimage
from skimage.filters import gaussian
from torch.autograd import Variable
from munkres import Munkres
from evaluation import adapted_rand
from orig_cremi_evaluation import voi
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from cremi.io import CremiFile
from cremi import Annotations, Volume

from skimage.measure import label
from matplotlib import pyplot as plt
import pdb
#rom skimage.color import label2rgb
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

def rondomwalker_seg(distance, threshold =2.0):
    if isinstance(distance, Variable):
        distance = distance.data
    if isinstance(distance, torch.FloatTensor) or isinstance(distance, torch.cuda.FloatTensor):
        distance = distance.cpu().numpy()
    #print(type(distance))
    distance = np.squeeze(distance)

    distance=gaussian(distance / np.max(distance), sigma=0.6, mode='reflect')
    markers = distance > 0.039
    markers = skimage.morphology.label(markers)
    seg_labels = random_walker(-distance, markers,\
                                       beta=25000, mode='cg_mg')
    #seg_labels = watershed(-distance, markers)

    #axes[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    return seg_labels



def convert_Variable2NumArray(tdata):
    if isinstance(tdata, Variable):
        tdata = tdata.data
    if isinstance(tdata, torch.FloatTensor) or isinstance(tdata, torch.cuda.FloatTensor):
        tdata = tdata.cpu().numpy()
    return tdata

def watershed_on_distance_and_skeleton(distance,skeleton, threshold =0.06):
    distance=convert_Variable2NumArray(distance)
    skeleton=convert_Variable2NumArray(skeleton)
    distance = np.squeeze(distance)
    skeleton =np.squeeze(skeleton)
    distance=gaussian(distance / float(np.max(distance)), sigma=0.6, mode='reflect')
    skeleton=gaussian(distance / float(np.max(skeleton)), sigma=0.6, mode='reflect')
    merged = distance +skeleton

    markers = merged > 0.25
    markers = skimage.morphology.label(markers)
    seg_labels = watershed(-distance, markers)
    #plt.imshow(merged)
    #plt.show()

    return seg_labels






def watershed_seg(distance,threshold = 0.06):
    if isinstance(distance, Variable):
        distance = distance.data
    if isinstance(distance, torch.FloatTensor) or isinstance(distance, torch.cuda.FloatTensor):
        distance = distance.cpu().numpy()
    #print(type(distance))
    distance = np.squeeze(distance)

    distance=gaussian(distance / float(np.max(distance)), sigma=0.6, mode='reflect')
    markers = distance > threshold
    markers = skimage.morphology.label(markers)
    seg_labels = watershed(-distance, markers)


    # image_max = ndi.maximum_filter(distance, size=10, mode='constant')
    # coordinates = peak_local_max(image_max, min_distance=15)

    # markers = np.zeros_like(distance)
    # pdb.set_trace()
    # markers[coordinates[:, 1], coordinates[:, 0]]=1
    # markers=label(markers)

    # print(np.unique(markers))
    #eg_labels = watershed(-distance, markers)

    #axes[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    return seg_labels


def watershed_seg2(distance_1,distance_fuse,threshold = 2.0):
    if isinstance(distance_1, Variable):
        distance_1 = distance_1.data
    if isinstance(distance_fuse, Variable)or isinstance(distance, torch.cuda.FloatTensor):
        distance_fuse = distance_fuse.data
    
    if isinstance(distance_fuse, torch.FloatTensor):
        distance_fuse = distance_fuse.cpu().numpy()
    if isinstance(distance_1, torch.FloatTensor) or isinstance(distance, torch.cuda.FloatTensor):
        distance_1 = distance_1.cpu().numpy()
    #distance_fuse = distance_fuse.cpu().numpy()
    distance_fuse = np.squeeze(distance_fuse)
    markers = distance_fuse > threshold
    markers = skimage.morphology.label(markers)

    #distance_1 = distance_1.cpu().numpy()
    distance_1 = np.squeeze(distance_1)
    


    seg_labels = watershed(-distance_1, markers)
    return seg_labels


def make_seg_submission(seg_valume_dict):
    submission_folder = 'submission'
    if not os.path.exists(submission_folder):
        os.makedirs(submission_folder)
    for name, seg_v in seg_valume_dict.iteritems():
        seg_v = seg_v.astype(np.uint64)
        neuron_ids = Volume(seg_v, resolution=(40.0, 4.0, 4.0), comment="Second submission in 2018")
        file = CremiFile(submission_folder+'/'+name + '.hdf', "w")
        file.write_neuron_ids(neuron_ids)


def replace_bad_slice_in_test(vol_data,set_name):
    replace_slice={}
    replace_slice['Set_A']={0:1,33:34,51:52,79:78,80:81,108:107,109:110,111:112}
    replace_slice['Set_B']={15:14,16:17,44:43,45:46,77:78}
    replace_slice['Set_C']={14:15,74:75,86:87}

    r_set = replace_slice[set_name]
    for idx,slice_d in enumerate(vol_data):
        if idx in r_set:
            vol_data[idx]=vol_data[r_set[idx]]
    return vol_data



if __name__ == '__main__':
    A =np.ones([22,1250,1250])
    B =np.ones([22,1250,1250])
    B[10:20,:,:]=6
    evalute_pred_dist_with_threshold(A,B)


