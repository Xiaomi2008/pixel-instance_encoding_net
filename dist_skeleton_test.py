import sys
#sys.path.append('../')
import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis, binary_erosion,binary_opening,binary_closing,binary_dilation, disk
import matplotlib.pyplot as plt
from skimage.color import label2rgb

from utils.EMDataset import slice_dataset
from utils.transform import *
from skimage.segmentation import watershed
import skimage
import pdb
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

data = np.squeeze(dset.__getitem__(80)['label'].cpu().numpy())[:320,:320]


aff_x = affinity(axis = -1,distance =6)
aff_y = affinity(axis = -2,distance =6)
compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0])==0).astype(np.int)
data = compute_boundary(data)


#print(data.shape)

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)

# skel_mask=ndimage.gaussian_filter(skel, sigma=l/(4.*n))
# skel = (skel > 0.4).astype(int)

#structure = disk(2, dtype=bool)
skel = binary_dilation(binary_dilation(skel))
#skel = binary_dilation(skel,structure)
dist_on_skel = distance * skel
#markers = skimage.morphology.label((dist_on_skel>1.5).astype(int))
markers = skimage.morphology.label(skel)
seg_labels = watershed(-distance, markers)

gx,gy   =  np.gradient(distance,1,1,edge_order =1)

# Distance to the background for pixels of the skeleton
#dist_on_skel = distance * skel

#dist_on_skel = binary_erosion(skel)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0,0].imshow(seg_labels, cmap=plt.cm.spectral, interpolation='nearest')
axs[0,0].axis('off')

axs[0,1].imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
axs[0,1].contour(data, [0.5], colors='w')
axs[0,1].axis('off')

axs[1,0].imshow(gx)
axs[1,0].axis('off')

axs[1,1].imshow(distance)
axs[1,1].axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()