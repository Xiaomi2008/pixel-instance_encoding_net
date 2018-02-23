import random
import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dis_transform
from skimage.morphology import medial_axis,binary_dilation
from scipy.ndimage.measurements import center_of_mass
import scipy.ndimage as nd
import pdb

class label_transform(object):
    def __init__(self, gradient     = True, 
                       distance     = True,
                       skeleton     = True,
                       objSizeMap   = False,
                       objCenterMap = False):
        self.gradient = gradient
        self.distance = distance
        self.objSizeMap =objSizeMap
        self.objCenterMap = objCenterMap
        self.skeleton = skeleton
        aff_x = affinity(axis = -1,distance =6)
        aff_y = affinity(axis = -2,distance =6)
        self.compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0])==0).astype(np.int)

    def __call__(self,*input):
        """
        Given input segmentation label of objects(neurons), 
        Generate 'Distance transform', 'gradient', 'size of object'
        labels.
        
        Args:
         2D numpy arrays, must be segmentation labels
        """

        
        #data = self.compute_boundary(data)
        # Compute the medial axis (skeleton) and the distance transform
        #skel, distance = medial_axis(data, return_distance=True)



        output =[]
        out_dict ={}
        for index,_input in enumerate(input):
            _input =np.squeeze(_input)
            boundary = self.compute_boundary(_input)
            if self.skeleton and self.distance:
                # Compute the medial axis (skeleton) and the distance transform
                skel, dt = medial_axis(boundary, return_distance=True)
                skel = binary_dilation(binary_dilation(skel))
                dist_on_skel = dt * skel
                sk = np.expand_dims(skel,0).astype(np.float)
            elif self.distance and not self.skeleton:
                dt = dis_transform(boundary)
            #markers = skimage.morphology.label((dist_on_skel>1.5).astype(int))
            #markers = skimage.morphology.label(skel)
            #seg_labels = watershed(-distance, markers)
            gx,gy   =  np.gradient(dt,1,1,edge_order =1)
            
            gx = np.expand_dims(gx,0).astype(np.float)
            gy = np.expand_dims(gy,0).astype(np.float)
            dt = np.expand_dims(dt,0).astype(np.float)
            

            out_dict['gradient'] = (gx,gy)
            
            if self.distance:
                out_dict['distance'] = dt
            
            if self.skeleton:
                out_dict['skeleton'] = sk
            
            if self.objSizeMap:
                sum_sizeMap =np.zeros_like(_input).astype(np.float)
                s_ids =np.unique(_input).tolist()
                for obj_id in s_ids:
                    obj_arr =  (_input == obj_id).astype(int)
                    obj_idx = obj_arr==1
                    sum_sizeMap[obj_idx]=(float(np.sum(obj_arr))/float(_input.size))*100.0

                sum_sizeMap = np.expand_dims(sum_sizeMap,0)
                out_dict['sizemap'] = sum_sizeMap
            output.append(out_dict)

        return tuple(output)


        # for idex,_input in enumerate(input):
        #     _input =np.squeeze(_input)
        #     s_ids =np.unique(_input).tolist()
        #     sum_gx =np.zeros_like(_input).astype(np.float)
        #     sum_gy =np.zeros_like(_input).astype(np.float)
        #     sum_dt =np.zeros_like(_input).astype(np.float)
        #     sum_sk =np.zeros_like(_input).astype(np.float)
        #     if self.objSizeMap:
        #         sum_sizeMap =np.zeros_like(_input).astype(np.float)
        #     for obj_id in s_ids:
        #         obj_arr =  (_input == obj_id).astype(int)
        #         dt      =  dis_transform(obj_arr)
                
        #         # Compute the medial axis (skeleton) and the distance transform
        #         #skel, dt = medial_axis(obj_arr, return_distance=True)
        #         # Distance to the background for pixels of the skeleton
        #         #dist_on_skel = dt * skel
                
        #         gx,gy   =  np.gradient(dt,dx,dy,edge_order =1)
        #         sum_gx+=gx
        #         sum_gy+=gy
        #         sum_dt+=dt
        #         sum_sk+=skel
        #         if self.objSizeMap:
        #             obj_idx = obj_arr==1
        #             sum_sizeMap[obj_idx]=(float(np.sum(obj_arr))/float(_input.size))*100.0
        #     # make it to 3D data (c,h,w)
        #     sum_gx = np.expand_dims(sum_gx,0)
        #     sum_gy = np.expand_dims(sum_gy,0)
        #     sum_dt = np.expand_dims(sum_dt,0)
        #     sum_sk = np.expand_dims(sum_sk,0)
            


        #     out_dict ={}
        #     if self.objSizeMap:
        #         sum_sizeMap = np.expand_dims(sum_sizeMap,0)
        #         out_dict['sizemap'] = sum_sizeMap
        #     out_dict['gradient'] = (sum_gx,sum_gy)
        #     if self.distance:
        #         out_dict['distance'] = sum_dt
        #     if self.skeleton:
        #         out_dict['skeleton'] = sum_sk
        #     output.append(out_dict)

        # return tuple(output)

def compute_boundary(x, axis = 1, distances = [2,4,8,16]):
    boudaries = np.stack(
                          [affinity(axis =axis, distance =dist)(x)[0] 
                                               for dist in distances],
                           axis=0
                        )
    return boudaries

class label_transform3D(object):
    def __init__(self, label_config = None,
                       distance     = True, 
                       boundary     = True,
                       gradient     = False,
                       skeleton     = False,
                       objCenterMap = False):
        #self.distance = distance
        #aff_x = affinity(axis = -1, distance =2)
        #aff_y = affinity(axis = -2, distance =2)
        #aff_z = affinity(axis = -3, distance =1)

        #self.compute_boundary = lambda x:((aff_x(x)[0]+aff_y(x)[0]+aff_z(x)[0])==0).astype(np.int)
        self.distance = distance
        self.boundary = boundary
        self.label_config = label_config
        self.affin_disX = [4,8,16]
        self.affin_disY = [4,8,16]
        self.affin_disZ = [1,2]
    
    def __compute_boudary__(self,_input):
        x =compute_boundary(_input,axis=-1, distances =self.affin_disX)
        y =compute_boundary(_input,axis=-2, distances =self.affin_disY)
        z =compute_boundary(_input,axis=-3, distances =self.affin_disZ)
        return x, y, z

    def __compute_2D_distance__(self, affinityX, affinityY):
        affinity2D = ((affinityX + affinityY)>0).astype(np.int)
        distance2D = np.stack(
                              [ 
                                np.stack( 
                                         [dis_transform(1-affinity[z]) for  z in range(affinity.shape[0])], axis=0
                                        )
                                for affinity in affinity2D
                              ],
                              axis = 0
                              )
        return distance2D

    def __compute_3D_distance__(self, affinityX, affinityY, affinityZ):
         ''' this channel in affinity represent different distance of affinity map
            where affinityZ  usually just have 1,2 pixel distance.
            so here ,  we will need sperately concatenate it with affiniyX and affinitY,
            currently, only use distance 2 of Z
          '''
         affinity2D = ((affinityX + affinityY)>0).astype(np.int)
         affinity3D =  np.stack([((each_affinity2D + affinityZ[1])>0).astype(np.int) for each_affinity2D in affinity2D],0)
         #affinity3D = ((affinityX + affinityY + affinityZ) >2).astype(np.int)
         distance3D = np.stack([ dis_transform(1-affinity_perCh) for affinity_perCh in affinity3D],axis=0)
         return distance3D

        # distance_per_ch_list =[]
        # for ch in affinity2D.shape[0]:
        #     affinity = affinity2D[ch]
        #     distance_2D = np.stack([dis_transform(affinity[z]) for  z in range(affinity.shape[0])],axis=0)
        #     distance_per_ch_list.append(distance_2D)

    def __call__(self,*input):
        """
        Given input segmentation label of objects(neurons), 
        Generate 'Distance transform'
        Note 3D version only support Distance transform
        labels.
        
        Args:
         3D numpy arrays, must be segmentation labels
        """

        output =[]
        for idex, _input in enumerate(input):
            out_dict ={}
            bx,by,bz =None, None, None
            if self.label_config:
                if 'affinityX' in self.label_config:
                    bx =compute_boundary(_input,axis=-1, distances =self.affin_disX)
                    out_dict['affinityX'] =bx
                if 'affinityY' in self.label_config:
                    by =compute_boundary(_input,axis=-2, distances =self.affin_disY)
                    out_dict['affinityY'] =by
                if 'affinityZ' in self.label_config:
                    bz =compute_boundary(_input,axis=-3, distances =self.affin_disZ)
                    out_dict['affinityZ'] =bz

                if 'distance2D' in self.label_config:
                    if bx is not None:
                        bx =compute_boundary(_input,axis=-1, distances =self.affin_disX)
                    if by is not None:
                        by =compute_boundary(_input,axis=-2, distances =self.affin_disY)
                    dist2D = self.__compute_2D_distance__(bx, by)
                    out_dict['distance2D'] =dist2D

                if 'distance3D' in self.label_config:
                    if bx is not None:
                        bx =compute_boundary(_input,axis=-1, distances =self.affin_disX)
                    if by is not None:
                        by =compute_boundary(_input,axis=-2, distances =self.affin_disY)
                    if bz is not None:
                        bz =compute_boundary(_input,axis=-3, distances =self.affin_disZ)
                    dist3D = self.__compute_3D_distance__(bx, by, bz)
                    out_dict['distance3D'] =dist3D
            else:
                if self.boundary:
                    bx,by,bz = self.__compute_boudary__(_input)
                    out_dict= {'affinityX':bx, 'affinityY':by, 'affinityZ':bz}
                if self.distance:
                    if not self.boundary:
                        bx,by,bz = self.__compute_boudary__(_input)
                    dist2D = self.__compute_2D_distance__(bx, by)
                    dist3D = self.__compute_3D_distance__(bx, by, bz)
                    out_dict['distance2D'] =dist2D
                    out_dict['distance3D'] =dist3D

            output.append(out_dict)

        return tuple(output)

    def output_labels(self):
        lable_channel_dict ={}
        if self.label_config:
            if 'affinityX' in self.label_config:
                lable_channel_dict['affinityX'] = len(self.affin_disX)
            if 'affinityY' in self.label_config:
                lable_channel_dict['affinityY'] = len(self.affin_disY)
            if 'affinityZ' in self.label_config:
                lable_channel_dict['affinityZ'] = len(self.affin_disZ)
            if 'distance2D' in self.label_config:
                lable_channel_dict['distance2D'] = len(self.affin_disX)
            if 'distance3D' in self.label_config:
                lable_channel_dict['distance3D'] = len(self.affin_disX)
        else:
            if self.boundary:
                lable_channel_dict ={'affinityX':len(self.affin_disX), 'affinityY':len(self.affin_disY),'affinityZ':len(self.affin_disZ)}
            if self.distance:
                lable_channel_dict['distance2D'] = len(self.affin_disX)
                lable_channel_dict['distance3D'] = len(self.affin_disX)

        return lable_channel_dict



           # return {'affinityX':len(self.affin_disX), 'affinityY':len(self.affin_disY),'affinityZ':len(self.affin_disZ),
           #     'distance2D':len(self.affin_disX),'distance3D':len(self.affin_disX)}
        

        #output =[]
        #dx,dy   = 1,1
        
        # for idex,_input in enumerate(input):
        #     _input =np.squeeze(_input)
            
        #     s_ids =np.unique(_input).tolist()
        #     sum_dt =np.zeros_like(_input).astype(np.float)
        #     for obj_id in s_ids:
        #         obj_arr =  (_input == obj_id).astype(int)
        #         dt      =  dis_transform(obj_arr)
        #         sum_dt+=dt
        #         if self.objSizeMap:
        #             obj_idx = obj_arr==1
        #     # make it to 4D data (c,h,w,d)
        #     sum_dt = np.expand_dims(sum_dt,0)
        #     out_dict ={}
        #     if self.distance:
        #         out_dict['distance'] = sum_dt
        #     output.append(out_dict)

        # return tuple(output)

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

class affinity(object):
    """
    Args:
        2D numpy arrays, must be segmentation labels
    """
    def __init__(self, axis=-1, distance=1):
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
            #zeros_pad_array = np.zeros_like(affinityMap)
            pad_array = np.ones_like(affinityMap)

            padslice1 = [slice(None)]*n_dim
            padslice2 = [slice(None)]*n_dim
            #padslice2 = [slice(None)]*n_dim
            #padslice1[self.axis] = slice(None,self.distance)
            padslice1[self.axis] = slice(None,1)
            padslice2[self.axis] = slice(-1,None)


            if self.distance % 2 == 0:
                affinityMap = np.concatenate([affinityMap[padslice1].repeat(self.distance/2,axis =self.axis),
                                          affinityMap,
                                          affinityMap[padslice2].repeat(self.distance/2,axis =self.axis)],
                                          self.axis)
            else:
                affinityMap = np.concatenate([affinityMap,
                                              affinityMap[padslice2].repeat(self.distance,axis =self.axis)],
                                              self.axis)
            # if self.distance % 2 == 0:
            #     affinityMap = np.concatenate([pad_array[padslice1].repeat(self.distance/2,axis =self.axis),
            #                               affinityMap,
            #                               pad_array[padslice2].repeat(self.distance/2,axis =self.axis)],
            #                               self.axis)
            # else:
            #     affinityMap = np.concatenate([affinityMap,
            #                                   pad_array[padslice2].repeat(self.distance,axis =self.axis)],
            #                                   self.axis)

            #                               
            #affinityMap = np.concatenate([pad_array[padslice1].repeat(self.distance,axis =self.axis),
            #                               affinityMap],
            #                               self.axis)
            
            #affinityMap = np.concatenate([zeros_pad_array[padslice1],affinityMap],self.axis)
            #
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
    def __init__(self, transform_list):
        """
        Args:
            number of tansform fuctions to be used  as below 
       
        """
        self.transform =transform_list
        #self.ZFlipFunc = 

    
    # def need_contrast_op(self,need = False):
    #     self.require_contrast =  need 
    
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
            #return  self.contrast_trans(*self.transform[func_idx](*input))
            transdata = self.transform[func_idx](*input)
            transdata = ZFlip()(*transdata) if random.randint(0,1) else transdata
            return transdata
    @property
    def name(self):
        out= []
        for trans in self.transform:
            out.append(trans.name)
        return '-'.join(out)

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
    @property
    def name(self):
        return 'VFlip'


class ZFlip(object):
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
            output.append(np.flip(_input,0).copy())
        return tuple(output)
    @property
    def name(self):
        return 'ZFlip'


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
    @property
    def name(self):
        return 'HFlip'

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
    @property
    def name(self):
        return 'Rot90'


class NRot90(object):
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
            output.append(np.rot90(_input,0).copy())
        return tuple(output)
    @property
    def name(self):
        return 'NRot90'



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
        #print(inputs)
        for idx, _input in enumerate(inputs):
            #print(_input.shape)
            channel_means = _input.mean(1).mean(1)
            channel_mean_array = np.zeros_like(_input)
            for i in range(len(_input)):
                channel_mean_array[i,:,:] = channel_means[i]
            _input = (_input - channel_mean_array) * self.value + channel_mean_array
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]
    @property
    def name(self):
        return 'Contrast'

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
    @property
    def name(self):
        return 'RandomContrast'

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
        value = np.random_choice(self.values, p=None)
        outputs = Contrast(value)(*inputs)
        return outputs
