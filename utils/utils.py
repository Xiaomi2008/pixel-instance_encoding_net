
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
def watershed_seg2D(distance):
    if isinstance(distance, Variable):
        distance = distance.data
    distance = distance.cpu().numpy()
    distance = np.squeeze(distance)
    markers = distance > 2
    markers = skimage.morphology.label(markers)
    seg_labels = watershed(-distance, markers)
    return seg_labels