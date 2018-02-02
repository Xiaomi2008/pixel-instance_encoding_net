import sys
#sys.path.append('../')
import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt

from utils.EMDataset import slice_dataset


def dataset(slice_axis):
            return slice_dataset(sub_dataset='Set_A',
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

data = dset.__getitem__(1)['label']

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0,0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
axs[0,0].axis('off')

axs[0,1].imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
axs[0,1].contour(data, [0.5], colors='w')
axs[0,1].axis('off')

axs[1,0].imshow(distance + skel*10 , cmap=plt.cm.spectral, interpolation='nearest')
axs[1,0].axis('off')

axs[1,1].imshow(distance  , cmap=plt.cm.spectral, interpolation='nearest')
axs[1,1].axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
plt.show()