import random
import torch
import numpy as numpy
from PIL import Image, ImageOps, ImageEnhance
from torhvision.transforms import functional as F
class RandomVerticalFlip(object):
def __call__(self,*input):
	"""
        Args:
            list 2d numpy array
        Returns:
            PIL Image: Randomly flipped data.
        """
        output =[]
        if random.random() < 0.5:
        	for idex,_input in enumerate(inputs):
        		output.apppend(np.fliplr(_input))
        return output


class RandomHorizontalFlip(object):
def __call__(self,*input):
	"""
        Args:
            input (list of PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        output =[]
        if random.random() < 0.5:
        	for idex,_input in enumerate(inputs):
        		output.apppend(np.flipud(_input))
        return output

class RandomRotate90(object):
def __call__(self,*input):
	"""
        Args:
            input (list of PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        output =[]
        if random.random() < 0.5:
        	for idex,_input in enumerate(inputs):
        		output.apppend(np.rot90(_input))
        return output
