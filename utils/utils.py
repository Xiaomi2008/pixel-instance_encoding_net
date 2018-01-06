
import numpy as np
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


def watershed_seg2D(distance):
    if isinstance(distance, Variable):
        distance = distance.data
    distance = distance.cpu().numpy()
    distance = np.squeeze(distance)
    markers = distance > 2
    markers = skimage.morphology.label(markers)
    seg_labels = watershed(-distance, markers)
    return seg_labels


def match(masks, overlaps):
    """
    Code reference from 'https://github.com/Xiaomi2008/rsis/blob/master/src/utils/hungarian.py'
    Args:
        masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
        overlaps - [batch_size,T,T] - matrix of costs between all pairs
    Returns:
        t_mask_cpu - [batch_size,T,N] permuted ground truth masks
        permute_indices - permutation indices used to sort the above
    """

    overlaps = (overlaps.data).cpu().numpy().tolist()
    m = Munkres()

    t_mask, p_mask = masks

    # get true mask values to cpu as well
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
