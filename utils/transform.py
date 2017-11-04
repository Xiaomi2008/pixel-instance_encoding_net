import random
import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dis_transform
import scipy.ndimage as nd

class gradient(object):
    def __call__(self,*input):
        output =[]
        for idex,_input in enumerate(input):
            _input =np.squeeze(_input)
            s_ids =np.unique(_input).tolist()
            sum_gx =np.zero_like(_input)
            sum_gy =np.zeros_like(_input)
            for obj_id in s_ids:
                obj_arr = (slice_lbs == obj_id).astype(int)
                dt  =  dis_transform(obj_arr)
                dx,dy   = 1,1
                gx,gy   = np.gradient(dt,dx,dy,edge_order =1)
                #gx-=np.min(gx)+0.01
                #gy-=np.min(gy)+0.01
                sum_gx+=gx
                sum_gy+=gy
                sum_dt+=dt
            output.append((sum_gx,sum_gy))
        return output
                #obj_idx = obj_arr==1
                #sum_obj_wt[obj_idx]=(float(image_size)/100.0)/float(np.sum(obj_arr))

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



class Contrast(object):
    """
    """
    def __init__(self, value):
        """
        Adjust Contrast of image.
        Contrast is adjusted independently for each channel of each image.
        For each channel, this Op computes the mean of the image pixels 
        in the channel and then adjusts each component x of each pixel to 
        (x - mean) * contrast_factor + mean.
        Arguments
        ---------
        value : float
            smaller value: less contrast
            ZERO: channel means
            larger positive value: greater contrast
            larger negative value: greater inverse contrast
        """
        self.value = value

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            channel_means = _input.mean(1).mean(2)
            channel_means = channel_means.expand_as(_input)
            _input = th.clamp((_input - channel_means) * self.value + channel_means,0,1)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class RandomContrast(object):

    def __init__(self, min_val, max_val):
        """
        Alter the Contrast of an image with a value randomly selected
        between `min_val` and `max_val`
        Arguments
        ---------
        min_val : float
            min range
        max_val : float
            max range
        """
        self.values = (min_val, max_val)

    def __call__(self, *inputs):
        value = random.uniform(self.values[0], self.values[1])
        outputs = Contrast(value)(*inputs)
        return outputs

class RandomChoiceContrast(object):

    def __init__(self, values, p=None):
        """
        Alter the Contrast of an image with a value randomly selected
        from the list of given values with given probabilities
        Arguments
        ---------
        values : list of floats
            contrast values to sampled from
        p : list of floats - same length as `values`
            if None, values will be sampled uniformly.
            Must sum to 1.
        """
        self.values = values
        self.p = p

    def __call__(self, *inputs):
        value = th_random_choice(self.values, p=None)
        outputs = Contrast(value)(*inputs)
        return outputs
