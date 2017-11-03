import random
import torch
import numpy as np

class random_transform(object):
    def __init__(self,*transform):
        """
        Args:
            number of tansform fuctions to be used  as below 
       
        """
        self.transform =transform
    def __call__(self,*input):
        """
        Args:
        list of 2d numpy array to be transformed
        return:
         randomly choise one transform to process the input data,
         or return data directly without process
        """
        func_idx = random.randint(0,len(self.transform))
        if func_idx ==len(self.transform):
            return input
        else:
            return  self.transform[func_idx](*input)

class VFlip(object):
    def __call__(self,*input):
        """
        Args:
            list 2d numpy array
        Returns:
            tuple: flipped data.
            """
        output =[]
        #rnd= random.random() < 0.5
        for idex,_input in enumerate(input):
            output.append(np.flip(_input,1).copy())
        return tuple(output)


class HFlip(object):
    def __call__(self,*input):
        """
        Args:
            input (list of nparray): Image to be flipped.
        Returns:
            tuple: flipped images.
        """
        output =[]
        #rnd= random.random() < 0.5
        for idex,_input in enumerate(input):
            output.append(np.flip(_input,2).copy())
        return tuple(output)

class Rot90(object):
    def __call__(self,*input):
        """
        Args:
            input (list of nparray): Image to be rotate.
        Returns:
            tuple: rotated images.
        """
        output =[]
        #rnd= random.random() < 0.5
        for idex,_input in enumerate(input):
            output.append(np.rot90(_input,2).copy())
        return tuple(output)