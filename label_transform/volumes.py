import pytoml as toml
import numpy as np
import h5py
from collections import namedtuple
import os
import pdb
import logging
import six
import random
DimOrder = namedtuple('DimOrder', ('X', 'Y', 'Z'))
class bounds_generator(six.Iterator):
    def __init__(self, volume_shape, subvolume_shape, seed = 1):
        self.volume_shape =volume_shape
        self.subvolume_shape =subvolume_shape
        self.seed = seed
        random.seed(self.seed)
    @property
    def shape(self):
		return self.subvolume_shape
    def __iter__(self):
        return self
    def __next__(self):
        bounds = []
        for v_dim,sub_dim in zip(self.volume_shape,self.subvolume_shape): 
            start= random.randrange(0,v_dim-sub_dim)
            end  = start + sub_dim
            bounds.append((start,end))
        return bounds


class SubvolumeGenerator(six.Iterator):
    def __init__(self,volume,bounds_generator):
        self.volume =volume
        self.bounds_generator =bounds_generator
    @property
    def shape(self):
        return self.bounds_generator.shape
    def __iter__(self):
        return self
    def __next__(self):
        return self.volume.get_subvolume(six.next(self.bounds_generator))


class Volume(object):
    DIM = DimOrder(Z=0, Y=1, X=2)
    def __init__(self, resolution, data_dict):
        self.data_dict = data_dict
    def local_coord_to_world(self, a):
        return a

    def world_coord_to_local(self, a):
        return a

    def world_mat_to_local(self, m):
        return m

    @property
    def shape(self):
        v0_key =self.data_dict.keys()[0]

        return tuple(self.world_coord_to_local(np.array(self.data_dict[v0_key].shape)))

    def _get_downsample_from_resolution(self, resolution):
        resolution = np.asarray(resolution)
        downsample = np.log2(np.true_divide(resolution, self.resolution))
        if np.any(downsample < 0):
            raise ValueError('Requested resolution ({}) is higher than volume resolution ({}). '
                             'Upsampling is not supported.'.format(resolution, self.resolution))
        if not np.all(np.equal(np.mod(downsample, 1), 0)):
            raise ValueError('Requested resolution ({}) is not a power-of-2 downsample of '
                             'volume resolution ({}). '
                             'This is currently unsupported.'.format(resolution, self.resolution))
        return downsample.astype(np.int64)

    def downsample(self, resolution):
        downsample = self._get_downsample_from_resolution(resolution)
        if np.all(np.equal(downsample, 0)):
            return self
        return DownsampledVolume(self, downsample)

    def partition(self, partitioning, partition_index):
        if np.array_equal(partitioning, np.ones(3)) and np.array_equal(partition_index, np.zeros(3)):
            return self
        return PartitionedVolume(self, partitioning, partition_index)

    def sparse_wrapper(self, *args):
        return SparseWrappedVolume(self, *args)

    def subvolume_bounds_generator(self, shape=None, label_margin=None):
        return self.SubvolumeBoundsGenerator(self, shape, label_margin)

    def subvolume_generator(self, bounds_generator=None, **kwargs):
        if bounds_generator is None:
            if not kwargs:
                raise ValueError('Bounds generator arguments must be provided if no bounds generator is provided.')
            bounds_generator = self.subvolume_bounds_generator(**kwargs)
        return SubvolumeGenerator(self, bounds_generator)

    def get_subvolume(self, bounds):
        # if bounds.start is None or bounds.stop is None:
        #     raise ValueError('This volume does not support sparse subvolume access.')

        def bounds2slice(bounds):
        	n_dim = len(bounds)
        	slices = [slice(None)]*n_dim
        	for i in range(n_dim):
        		slices[i] = slice(bounds[i][0],bounds[i][1])
        	return slices
        b_slices = bounds2slice(bounds)
        subvolumes = { name:data[b_slices] 
                       for name, data in self.data_dict.iteritems()}
        return subvolumes

class HDF5Volume(Volume):
    """A volume backed by data views to HDF5 file arrays.

    Parameters
    ----------
    orig_file : str
        Filename of the HDF5 file to load.
    image_dataaset : str
        Full dataset path including groups to the raw image data array.
    label_dataset : str
        Full dataset path including groups to the object label data array.
    """
    @staticmethod
    def from_toml(filename):
        from keras.utils.data_utils import get_file
        volumes = {}
        with open(filename, 'rb') as fin:
        	ld = toml.load(fin).get('local_data',None)
                print(filename)
        	data_dir =ld['data_dir']
        	if not os.path.exists(data_dir):
        		os.mkdir(data_dir)
        with open(filename, 'rb') as fin:
        	datasets = toml.load(fin).get('dataset', [])
        	print ('len is {}'.format(len(datasets)))
        	for dataset in datasets:
        		hdf5_file = dataset['hdf5_file']
        		local_file = data_dir + '/'+ hdf5_file
        		if not os.path.exists(local_file):
        			hdf5_file = get_file(hdf5_file, dataset['download_url'], 
        								md5_hash=dataset.get('download_md5', None), 
        								cache_subdir='', 
        								cache_dir=data_dir)
        		dataset_dict ={data['name']:data['path'] for data in dataset.get('data',None)}
        		# for data in in all_data:
        		#     data_dict[data['name']]=data['path']
                mask_bounds ='dummy'
                volume = HDF5Volume(local_file,dataset_dict,mask_bounds=mask_bounds)
                volumes[dataset['name']] = volume

        return volumes

    @staticmethod
    def write_file(filename, **kwargs):
        h5file = h5py.File(filename, 'w')
        config = {'hdf5_file': filename}
        channels = ['image', 'label', 'mask','gradX','gradY','gradZ','distTF'
                    'affinX1','affinX3','affinX5','affinX7','affinX13','affinX20',
                    'affinY1','affinY3','affinY5','affinY7','affinX13','affinX20',
                    'affinZ1','affinZ3']
        default_datasets = {
            'image': 'volumes/raw',
            'label': 'volumes/labels/neuron_ids',
            'mask': 'volumes/labels/mask',
            'gradX': 'transformed_label/directionX',
            'gradY': 'transformed_label/directionY',
            'gradZ': 'transformed_label/directionZ',
            'distTF': 'transformed_label/distance',
            'affinX1': 'affinity_map/x1',
            'affinX3': 'affinity_map/x3',
            'affinX5': 'affinity_map/x5',
            'affinX7': 'affinity_map/x7',
            'affinX13': 'affinity_map/x13',
            'affinX20': 'affinity_map/x20',
            'affinY1': 'affinity_map/y1',
            'affinY3': 'affinity_map/y3',
            'affinY5': 'affinity_map/y5',
            'affinY7': 'affinity_map/y7',
            'affinY13': 'affinity_map/y13',
            'affinY20': 'affinity_map/y20',
            'affinZ1': 'affinity_map/z1',
            'affinZ3': 'affinity_map/z3',
        }
        for channel in channels:
            data = kwargs.get('{}_data'.format(channel), None)
            dataset_name = kwargs.get('{}_dataset'.format(channel), default_datasets[channel])
            if data is not None:
                dataset = h5file.create_dataset(dataset_name, data=data, dtype=data.dtype)
                #dataset.attrs['resolution'] = resolution
                config['{}_dataset'.format(channel)] = dataset_name

        h5file.close()

        return config
    def __init__(self, orig_file, dataset_dict ,mask_bounds=None):
        logging.debug('Loading HDF5 file "{}"'.format(orig_file))
        self.file = h5py.File(orig_file, 'r')
        self.resolution = None
        self._mask_bounds = tuple(map(np.asarray, mask_bounds)) if mask_bounds is not None else None
        #print(self.file['transformed_label'].keys())
        #print dataset_dict
        for name, data in dataset_dict.iteritems():
            d = np.array(self.file[data])

        self.data_dict ={name:np.array(self.file[data]) for name, data in dataset_dict.iteritems()}
        #print(self.data_dict[self.data_dict.keys()[0]].shape)

def run_test():
    print('read')
    file_name='../conf/cremi_datasets.toml'
    VS = HDF5Volume.from_toml(file_name)
    V1=VS[VS.keys()[0]]
    bounds_gen=bounds_generator(V1.shape,[10,320,320])
    sub_vol_gen =SubvolumeGenerator(V1,bounds_gen)
    for i in xrange(200):
        C = six.next(sub_vol_gen);
        for name,data in C.iteritems():
            print('{} shape = {}'.format(name,data.shape))



if __name__ == "__main__":
	run_test()
