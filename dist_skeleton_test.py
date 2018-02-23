import sys
#sys.path.append('../')
import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis, binary_erosion,binary_opening, \
                               binary_closing,binary_dilation, disk, \
                               skeletonize,skeletonize_3d
from scipy.ndimage.morphology import distance_transform_edt as dis_transform
import matplotlib.pyplot as plt
from skimage.color import label2rgb
#from skimage.morphology import skeletonize, skeletonize_3d

from utils.EMDataset import slice_dataset
from utils.transform import *
from skimage.segmentation import watershed
import skimage
import pdb
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi

def dataset(slice_axis):
            return slice_dataset(sub_dataset='All',
                                     subtract_mean=True,
                                     split='valid',
                                     slices=1,
                                     slice_axis = 0,
                                     data_config= 'conf/cremi_datasets.toml')

def microstructure(l=256):
    """
    Synthetic binary data: binary microstructure with blobs.

    Parameters
    ----------

    l: int, optional
        linear size of the returned image

    """
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask_outer = (x - l/2)**2 + (y - l/2)**2 < (l/2)**2
    mask = np.zeros((l, l))
    generator = np.random.RandomState(3)
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l/(4.*n))
    return mask > mask.mean()

#data = microstructure(l=128)
dset = dataset(slice_axis=0)
dset.set_current_subDataset('Set_B')

data_list = [np.squeeze(dset.__getitem__(i)['label'].cpu().numpy())[:1248,:1248] for i in range(80,125)]
gt_seg =np.stack(data_list,0)


aff_x = affinity(axis = -1,distance =2)
aff_y = affinity(axis = -2,distance =2)
aff_z = affinity(axis = -3,distance =1)
#compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0] +aff_z(x)[0])==0).astype(np.int)
compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0])==0).astype(np.int)
affinity = compute_boundary(gt_seg)


#gx,gy   =  np.gradient(dt,1,1,edge_order =1)


#print(data.shape)

# Compute the medial axis (skeleton) and the distance transform
#skel, distance = medial_axis(affinity, return_distance=True)



distance =  np.stack([dis_transform(affinity[i]) for i in range(len(affinity))],0)

#distance = dis_transform(affinity)

#skel =skeletonize_3d(affinity)

# skel_mask=ndimage.gaussian_filter(skel, sigma=l/(4.*n))
# skel = (skel > 0.4).astype(int)

#structure = disk(2, dtype=bool)
#skel = binary_dilation(binary_dilation(skel))
#skel = binary_dilation(skel,structure)
dist_on_skel = distance #* skel
#markers = skimage.morphology.label((dist_on_skel>1.5).astype(int))
#markers = skimage.morphology.label(skel)

gt_seg =gt_seg[1:2]
distance = distance[1:2]
affinity =affinity[1:2]
seg2 = (1-affinity) * gt_seg

markers = skimage.morphology.label(affinity)
seg_labels = watershed(-distance, markers)


print('dt = {}'.format(distance.shape))
#gz,gx,gy =  np.gradient(distance,1,1,1, edge_order =1)
gx,gy =  np.gradient(distance[0],1,1, edge_order =1)

# Distance to the background for pixels of the skeleton
#dist_on_skel = distance * skel12

#dist_on_skel = binary_erosion(skel)
#
#
#
#
#seg_labels[seg_labels ==28]=79
#seg_labels[seg_labels ==93]=79

arand = adapted_rand(seg_labels.astype(np.int), gt_seg)
split, merge = voi(seg_labels.astype(np.int), gt_seg)



print('rand , voi Merg, Split ={:.3f}, ({:.3f},{:.3f})'.format(arand,merge,split))
slice_idx =0

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0,0].imshow(seg_labels[slice_idx], cmap=plt.cm.spectral, interpolation='nearest')
axs[0,0].title.set_text('seg_waterseg')
axs[0,0].axis('off')

axs[0,1].imshow(gt_seg[slice_idx], cmap=plt.cm.spectral, interpolation='nearest')
#axs[0,1].contour(data[slice_idx], [0.5], colors='w')
axs[0,1].title.set_text('gt')
axs[0,1].axis('off')

axs[1,0].imshow(gy)
axs[1,0].title.set_text('gradient')
axs[1,0].axis('off')

axs[1,1].imshow(distance[slice_idx])
axs[1,1].title.set_text('distance')
axs[1,1].axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()