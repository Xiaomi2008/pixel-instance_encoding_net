import random
import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dis_transform
import scipy.ndimage as nd

class label_transform(object):
    def __init__(self, gradient = True, 
                       distance = True,
                       objSizeMap =False):
        self.gradient = gradient
        self.distance = distance
        self.objSizeMap =objSizeMap

    def __call__(self,*input):
        """
        Args:
         2D numpy arrays, must be segmentation labels
        """
        output =[]
        dx,dy   = 1,1
        for idex,_input in enumerate(input):
            _input =np.squeeze(_input)
            s_ids =np.unique(_input).tolist()
            sum_gx =np.zeros_like(_input).astype(np.float)
            sum_gy =np.zeros_like(_input).astype(np.float)
            sum_dt =np.zeros_like(_input).astype(np.float)
            if self.objSizeMap:
                sum_sizeMap =np.zeros_like(_input).astype(np.float)
            for obj_id in s_ids:
                obj_arr =  (_input == obj_id).astype(int)
                dt      =  dis_transform(obj_arr)
                gx,gy   =  np.gradient(dt,dx,dy,edge_order =1)
                sum_gx+=gx
                sum_gy+=gy
                sum_dt+=dt
                if self.objSizeMap:
                    obj_idx = obj_arr==1
                    sum_sizeMap[obj_idx]=(float(_input.size)/300.0)/float(np.sum(obj_arr))
            # make it to 3D data (c,h,w)
            sum_gx = np.expand_dims(sum_gx,0)
            sum_gy = np.expand_dims(sum_gy,0)
            sum_dt = np.expand_dims(sum_dt,0)
            if self.objSizeMap:
                sum_sizeMap = np.expand_dims(sum_sizeMap,0)
            out_dict ={}
            out_dict['gradient'] = (sum_gx,sum_gy)
            if self.distance:
                out_dict['distance'] = sum_dt
            if self.objSizeMap:
                out_dict['sizemap'] = sum_sizeMap
            output.append(out_dict)
        return tuple(output)
class affinity(object):
    """
    Args:
        2D numpy arrays whuch must be segmentation labels
    """
    def __init__(self, axis=3,distance=1):
        self.distance = distance
        self.axis     = axis
    def __call__(self,*input):
        for idex,_input in enumerate(input):
            _shape =  _input.shape
            #print _shape
            n_dim  =  _input.ndim
            slice1 = [slice(None)]*n_dim
            slice2 = [slice(None)]*n_dim
            slice1[self.axis] = slice(self.distance,None)
            slice2[self.axis] = slice(None,-self.distance)
            affinityMap= (abs(_input[slice1] - _input[slice2]) > 0 ).astype(np.int32)
            if self.axis ==-1:
                zeros_pad_array = np.zeros([_shape[0], _shape[-2],self.distance])
            elif self.axis ==-2:
                zeros_pad_array = np.zeros([_shape[0], self.distance, _shape[-1]])
            #print zeros_pad_array.shape
            #print affinityMap.shape
            affinityMap = np.concatenate([zeros_pad_array,affinityMap],self.axis)
        return affinityMap


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
            if idx == 0:
                channel_means = np.mean(_input)
                _input = np.clip(((_input - channel_means) * self.value + channel_means), a_min = 0, a_max = 255)
                #_input -= np.min(_input)
                #_input = (_input/np.max(_input)) * 255
                outputs.append(_input)
            else:
                outputs.append(_input)
        return outputs 

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
