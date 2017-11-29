import random
import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dis_transform
from scipy.ndimage.measurements import center_of_mass
import scipy.ndimage as nd
import math
import pdb
from affine_utils import *

class label_transform(object):
    def __init__(self, gradient     = True, 
                       distance     = True,
                       objSizeMap   = False,
                       objCenterMap = False):
        self.gradient = gradient
        self.distance = distance
        self.objSizeMap =objSizeMap
        self.objCenterMap = objCenterMap

    def __call__(self,*input):
        """
        Given input segmentation label of objects(neurons), 
        Generate 'Distance transform', 'gradient', 'size of object'
        labels.
        
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
            


            out_dict ={}
            if self.objSizeMap:
                sum_sizeMap = np.expand_dims(sum_sizeMap,0)
                out_dict['sizemap'] = sum_sizeMap
            out_dict['gradient'] = (sum_gx,sum_gy)
            if self.distance:
                out_dict['distance'] = sum_dt
            # if self.objSizeMap:
            #     out_dict['sizemap'] = sum_sizeMap

            output.append(out_dict)

        return tuple(output)

class objCenterMap(object):
    def __call__(self, *input):
        out_list = []
        aff_x = affinity(axis = -1,distance =2)
        aff_y = affinity(axis = -2,distance =2)

        compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0])==0).astype(np.int)
        for idex, _input in enumerate(input):
            _input =np.squeeze(_input)
            s_ids =np.unique(_input).tolist()
            x_center =np.zeros_like(_input).astype(np.float)
            y_center =np.zeros_like(_input).astype(np.float)
            data = compute_boundary(_input)
            label,f =nd.label(data)
            #print np.unique(label)
            centers = center_of_mass(data, labels=label,index=np.unique(label)[1:])
            #print centers
            for i, (cx,cy) in enumerate(centers):
                obj_arr = (label == i+1)
                x_center[obj_arr] = cx / obj_arr.shape[0]
                y_center[obj_arr] = cy / obj_arr.shape[1]
            x_center = np.expand_dims(x_center,0)
            y_center = np.expand_dims(y_center,0)
            out_list.append((x_center,y_center))
        return tuple(out_list)

def my_center_of_mass(x):
    pass

class affinity(object):
    """
    Args:
        2D numpy arrays whuch must be segmentation labels
    """
    def __init__(self, axis=-1,distance=1):
        self.distance = distance
        self.axis     = axis
    def __call__(self,*input):
        out_list  =[]
        for idex,_input in enumerate(input):
            _shape =  _input.shape
            #print _shape
            n_dim  =  _input.ndim
            slice1 = [slice(None)]*n_dim
            slice2 = [slice(None)]*n_dim
            slice1[self.axis] = slice(self.distance,None)
            slice2[self.axis] = slice(None,-self.distance)
            affinityMap= (abs(_input[slice1] - _input[slice2]) > 0 ).astype(np.int32)
            zeros_pad_array = np.zeros_like(affinityMap)

            padslice1 = [slice(None)]*n_dim
            #padslice2 = [slice(None)]*n_dim
            padslice1[self.axis] = slice(None,self.distance)

            affinityMap = np.concatenate([zeros_pad_array[padslice1],affinityMap],self.axis)


            # if self.axis ==-1:
            #     zeros_pad_array = np.zeros([_shape[0], _shape[-2],self.distance])
            # elif self.axis ==-2:
            #     zeros_pad_array = np.zeros([_shape[0], self.distance, _shape[-1]])
            # print zeros_pad_array.shape, affinityMap.shape,self.axis
            # affinityMap = np.concatenate([zeros_pad_array,affinityMap],self.axis)
            out_list.append(affinityMap)
        return tuple(out_list)
        #return affinityMap



class random_transform(object):
    def __init__(self,*transform):
        """
        Args:
            number of tansform fuctions to be used  as below 
       
        """
        self.transform =transform
    def __call__(self):
        """
        Args:
        list of 2d numpy array to be transformed
        return:
         randomly choise one transform to process the input data,
         or return data directly without process
        """
        print("call transform")
        func_idx = random.randint(0,len(self.transform))
        if func_idx ==len(self.transform):
            out1 = lambda *x: x
            out2 = 'identity'
        else:
            out1 = self.transform[func_idx]
            out2 = self.transform[func_idx].category
        return out1, out2


class VFlip(object):
    def __init__(self):
        self.category = 'regular'

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
    def __init__(self):
        self.category = 'regular'

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
    def __init__(self):
        self.category = 'regular'

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
        self.category = 'regular'

    def __call__(self, *inputs):
        output = []
        for idx, _input in enumerate(inputs):
            if idx == 0:
                channel_means = np.mean(_input)
                _input = np.clip(((_input - channel_means) * self.value + channel_means), a_min = 0, a_max = 255)
                #_input -= np.min(_input)
                #_input = (_input/np.max(_input)) * 255
                output.append(_input)
            else:
                output.append(_input)
        return tuple(output)

 class Shear(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        self.value = value
        self.interp = interp
        self.lazy = lazy
        self.category = 'affine'

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = (math.pi * self.value) / 180
        shear_matrix = torch.FloatTensor([[1, -math.sin(theta), 0],
                                        [0, math.cos(theta), 0],
                                        [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            output = []
            for idx, _input in enumerate(inputs):
                _input = torch.from_numpy(_input)
                input_tf = th_affine2d(_input,
                                       shear_matrix,
                                       mode=interp[idx],
                                       center=True)
                input_tf = input_tf.numpy()
                output.append(input_tf)
                # output = output.numpy()
            # return output if idx > 1 else output[0]
            return tuple(output)

class Zoom(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float
            Fractional zoom.
            =1 : no zoom
            >1 : zoom-in (value-1)%
            <1 : zoom-out (1-value)%

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy: boolean
            If true, just return transformed
        """

        if not isinstance(value, (tuple,list)):
            value = (value, value)
        self.value = value
        self.interp = interp
        self.lazy = lazy
        self.category = 'affine'

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        zx, zy = self.value
        zoom_matrix = th.FloatTensor([[zx, 0, 0],
                                      [0, zy, 0],
                                      [0, 0,  1]])        

        if self.lazy:
            return zoom_matrix
        else:
            output = []
            for idx, _input in enumerate(inputs):
                _input = torch.from_numpy(_input)
                input_tf = th_affine2d(_input,
                                       zoom_matrix,
                                       mode=interp[idx],
                                       center=True)
                input_tf = input_tf.numpy()
                output.append(input_tf)
            # return output if idx > 1 else output[0]
            return tuple(output)
        
class Rotate(object):

    def __init__(self, 
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']

        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.value = value
        self.interp = interp
        self.lazy = lazy
        self.category = 'affine'

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rotation_matrix = th.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                          [math.sin(theta), math.cos(theta), 0],
                                          [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            output = []
            for idx, _input in enumerate(inputs):
                _input = torch.from_numpy(_input)
                input_tf = th_affine2d(_input,
                                       rotation_matrix,
                                       mode=interp[idx],
                                       center=True)
                input_tf = input_tf.numpy()
                output.append(input_tf)
            return output
            # return output if idx > 1 else output[0]
        
def _blur_image(image, H):
    # break image up into its color components
    size = image.shape
    # imr = image[0,:,:]
    # img = image[1,:,:]
    # imb = image[2,:,:]

    # compute Fourier transform and frequqnecy spectrum
    Fim1 = np.fft.fftshift(np.fft.fft2(image))
    # Fim1r = np.fft.fftshift(np.fft.fft2(imr))
    # Fim1g  = np.fft.fftshift(np.fft.fft2(img))
    # Fim1b  = np.fft.fftshift(np.fft.fft2(imb))
    
    # Apply the lowpass filter to the Fourier spectrum of the image
    filtered_image = np.multiply(H, Fim1)
    # filtered_imager = np.multiply(H, Fim1r)
    # filtered_imageg = np.multiply(H, Fim1g)
    # filtered_imageb = np.multiply(H, Fim1b)
    
    newim = np.zeros(size)

    # convert the result to the spatial domain.
    newim = np.absolute(np.real(np.fft.ifft2(filtered_image)))
    # newim[1,:,:] = np.absolute(np.real(np.fft.ifft2(filtered_imageg)))
    # newim[2,:,:] = np.absolute(np.real(np.fft.ifft2(filtered_imageb)))

    return newim.astype('uint8')

def _butterworth_filter(rows, cols, thresh, order):
    # X and Y matrices with ranges normalised to +/- 0.5
    array1 = np.ones(rows)
    array2 = np.ones(cols)
    array3 = np.arange(1,rows+1)
    array4 = np.arange(1,cols+1)

    x = np.outer(array1, array4)
    y = np.outer(array3, array2)

    x = x - float(cols/2) - 1
    y = y - float(rows/2) - 1

    x = x / cols
    y = y / rows

    radius = np.sqrt(np.square(x) + np.square(y))

    matrix1 = radius/thresh
    matrix2 = np.power(matrix1, 2*order)
    f = np.reciprocal(1 + matrix2)

    return f


class Blur(object):
    """
    Blur an image with a Butterworth filter with a frequency
    cutoff matching local block size
    """
    def __init__(self, threshold, order=5):
        """
        scramble blocksize of 128 => filter threshold of 64
        scramble blocksize of 64 => filter threshold of 32
        scramble blocksize of 32 => filter threshold of 16
        scramble blocksize of 16 => filter threshold of 8
        scramble blocksize of 8 => filter threshold of 4
        """
        self.threshold = threshold
        self.order = order
        self.category = 'blur'

    def __call__(self, *inputs):
        """
        inputs should have values between 0 and 255
        """
        output = []
        for idx, _input in enumerate(inputs):
            if idx == 0:
                rows = _input.shape[0]
                cols = _input.shape[1]
                fc = self.threshold # threshold
                fs = 128.0 # max frequency
                n  = self.order # filter order
                fc_rad = (fc/fs)*0.5
                H = _butterworth_filter(rows, cols, fc_rad, n)
                _input_blurred = _blur_image(_input.astype('uint8'), H)
                # _input_blurred = np.asarray(_input_blurred)
                # _input_blurred = th.from_numpy(_input_blurred).float()
                output.append(_input_blurred)
                # output = np.asarray(_input_blurred)
            else:
                output.append(_input)

        # return output if idx > 1 else output[0]
        return tuple(output)

class RandomBlur(object):
    """
    Blur an image at boundary 
    """
    def __init__(self, num_patch = 5, patch_size = 10, threshold = 32):
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.threshold = threshold
        self.category = 'randomBlur'
        self.transform = Blur(self.threshold)

    def __call__(self, *inputs):
        output = []
        affineX=affinity(axis=-1,distance =1)
        affineY=affinity(axis=-2,distance =1)
        data = inputs[0]
        x_size = data.shape[0]
        y_size = data.shape[1]
        seg_label = inputs[1]
        im = data[0].copy()
        affinMap = ((affineX(seg_label)[0] + affineY(seg_label)[0])>0).astype(np.int)
        affinity_set = np.where(affinMap[0,self.patch_size:x_size-self.patch_size, self.patch_size:y_size-self.patch_size] == 1)
        loc = np.random.choice(affinity_set[0].shape[0], self.num_patch, replace = False)
        x_loc, y_loc = affinity_set[0][loc] + self.patch_size, affinity_set[1][loc] + self.patch_size
        for i in range(0, self.num_patch):
            _im = im[x_loc[i]-self.patch_size:x_loc[i]+self.patch_size, y_loc[i]-self.patch_size:y_loc[i]+self.patch_size]
            new_im = self.transform(_im)
            im[x_loc[i]-self.patch_size:x_loc[i]+self.patch_size, y_loc[i]-self.patch_size:y_loc[i]+self.patch_size] = new_im[0]
        output.append(np.reshape(im, data.shape)) 
        output.append(seg_label)
        return tuple(output)

        
