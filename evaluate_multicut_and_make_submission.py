from skimage.segmentation import relabel_sequential
from utils.utils import replace_bad_slice_in_test as replace_bad_slice, make_seg_submission
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi
import h5py, numpy as np
import pdb

set_names= ['Set_A','Set_B','Set_C']
def read_raw_image_data(set_name,subset_name):
    data_path ={}
    data_path['test'] ={'Set_A':'data/sample_A+_20160601.hdf',
                        'Set_B':'data/sample_B+_20160601.hdf',
                        'Set_C':'data/sample_C+_20160601.hdf'}

    data_path['valid'] ={'Set_A':'data/sample_A_20160501.hdf',
                        'Set_B':'data/sample_B_20160501.hdf',
                        'Set_C':'data/sample_C_20160501.hdf'}

    hd5_file = data_path[set_name][subset_name]
    h5=h5py.File(hd5_file,'r')
    raw_h5_path='volumes/raw'
    lb_h5_path ='volumes/labels/neuron_ids'
    raw_im = np.array(h5[raw_h5_path]).astype(np.int)
    if set_name =='valid':
        lbs    = np.array(h5[lb_h5_path]).astype(np.int)
        h5.close()
    else:
        lbs =None

    return raw_im, lbs

def read_predict_segmentation(set_name, subset_name):
    data_path={}
    data_path['test'] ={'Set_A':'tempdata/multicut_result/test/Set_A_mcut_with2Dseg_amazing.h5',#,multi_cut_valid_setA_thresd0.4.h5
                        'Set_B':'tempdata/multicut_result/test/Set_B_mcut_with2Dseg.h5',
                        'Set_C':'tempdata/multicut_result/test/Set_C_mcut_with2Dseg.h5'}

    data_path['valid'] ={'Set_A':'tempdata/multicut_result/valid/Set_A_mcut_with2Dseg_amazing_0.25.h5', #Set_A_mcut_with2Dseg_amazing.h5',
                         'Set_B':'tempdata/multicut_result/valid/Set_B_with_2Dseg_beta_0.4.h5',
                         'Set_C':'tempdata/multicut_result/valid/Set_C_mcut_with2Dseg.h5'}

    hd5_file = data_path[set_name][subset_name]
    h5=h5py.File(hd5_file,'r')
    print(h5.keys())
    seg_h5_path =h5.keys()[0]
    seg = np.array(h5[seg_h5_path]).astype(np.uint)
    h5.close()
    return seg


def evaluation(subset_name):
    seg=read_predict_segmentation('valid', subset_name)
    _, lbs=read_raw_image_data('valid',subset_name)
    #pdb.set_trace()
    np.squeeze(seg)
    arand =adapted_rand(seg,lbs)
    split,merge = voi(seg,lbs)

    print('{} --- arand : (split,merge) {}:({},{})'.format(subset_name, arand,split,merge))

if __name__=='__main__':
    evaluation('Set_A')

