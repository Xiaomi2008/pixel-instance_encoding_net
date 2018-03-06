import sys, os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import watershed_seg, rondomwalker_seg, watershed_on_distance_and_skeleton, make_seg_submission
from utils.slice_connector import Simple_MaxCoverage_3DSegConnector, NN_slice_3DSegConnector
import pdb
from skimage.segmentation import relabel_sequential
from utils.EMDataset import slice_dataset
from utils.utils import replace_bad_slice_in_test as replace_bad_slice
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi
import torch
my_dpi=96
set_names= ['Set_A','Set_B','Set_C']
data_names=['distance','final']



def segment(hd5_file, set_name, raw_im,data_name_for_seg='distance', rep_bad_slice=False,connect_2d=True):
    h5=h5py.File(hd5_file)
    print(h5.keys())
    seg_dict ={}
    h5_path=set_name+'_'+data_name_for_seg
    distance=np.array(h5[h5_path])
    h5.close()
    #NNS=NN_slice_3DSegConnector()
    #get_rawim_or_labels(task_set=set_to_process,subset_name,data_set):
    seg_vol = np.zeros_like(distance)
    data = replace_bad_slice(data,set_name) \
                              if rep_bad_slice \
                              else data
    # data = replace_bad_slice(data,set_name) \
    #                           if rep_bad_slice \
    #                           else data

    for idx, slice_d in enumerate(distance):
        print('segmenting slice {}'.format(idx))
        seg_vol[idx] = watershed_seg(slice_d,threshold=0.1)
    if not connect_2d:
        for idx,seg in emumerate(seg_vol):
            seg_vol[i]+=idx*3000
        seg3D =seg_vol
    else:
        seg3D=connect_2d_slice(raw_im,seg_vol)
        seg3D, _ , _ =relabel_sequential(seg3D)
    return seg3D

def segment_all_and_makeSubmission(hd5_file,data_name_for_seg ='distance',\
                                  im_dir=None,\
                                  task_set='test',
                                  save_seg_to_img=False, \
                                  rep_bad_slice=True,
                                  evaluation =True):
    seg_dict ={}
    for set_n in set_names:
        raw_im=get_rawim_or_labels(task_set=task_set,subset_name=set_n,data_set='image')
        raw_im = replace_bad_slice(raw_im,set_n) \
                              if rep_bad_slice \
                              else raw_im
        seg3D=segment(hd5_file,set_n,raw_im, data_name_for_seg,rep_bad_slice)
        if evaluation and task_set =='valid':
            gt_label =get_rawim_or_labels(task_set=task_set,subset_name=set_n,data_set='label')
            gt_label = gt_label[0:len(seg3D)]
            arand=adapted_rand(seg3D,gt_label)
            split,merge =voi(seg3D,gt_label)
            print('arand {} ,(split,merge) =({},{})'.format(arand,split,merge))
        seg_dict[set_n]=seg3D
        
        if save_seg_to_img:
            assert im_dir
            im_save_dir=im_dir+'_'+set_n
            if not os.path.exists(im_save_dir):
                os.mkdir(im_save_dir)
            save_seg3D_to_image(seg3D,im_save_dir, data_name_for_seg)
        #pdb.set_trace()
        #make_seg_submission(seg_dict)
def guidedSegmentation(hd5_file, task_set='test',rep_bad_slice=True):
    h5=h5py.File(hd5_file)
    print(h5.keys())
    seg_dict ={}
    set_name = 'Set_A'
    data_name_for_seg ='distance'
    h5_path=set_name+'_'+data_name_for_seg
    distance=np.array(h5[h5_path])
    h5.close()
    for idx, slice_d in enumerate(distance[0:70]):
        print('segmenting slice {}'.format(idx))
        next_d =distance[idx+1]
        seg1 = watershed_seg(slice_d,threshold=0.1)
        #seg2 = watershed_seg(next_d,threshold=0.1)
        pts,seg3 = show2seg(seg1,seg1)
        a=merge_clicks(pts,seg3)
        show1seg(a)
        #plt.ginput(1)
        #click_markers =plt.ginput(n=2, timeout=-1)

def merge_clicks(pts, seg):
    print pts
    for (x,y) in pts:
        sid = seg[int(x),int(y)]
        print('sid:{}'.format(sid))
    sids= [seg[int(x),int(y)] for (x,y)in pts]

    assgined_id=sids[0]
    for sid in sids:
        seg[seg==sid] =assgined_id
    return seg


def show1seg(a):
     a_seq, _ , _ =relabel_sequential(a)
     a_seq=np.random.permutation(a_seq.max() + 1)[a_seq]
     plt.imshow(a_seq,cmap='spectral')
     plt.show()
def show2seg(a,b):
     #fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
     # axes[0].imshow(a,cmap='spectral')
     # axes[0].axis('off')
     # axes[0].margins(0, 0)
     # axes[0].set_title('upper')
     a_seq, _ , _ =relabel_sequential(a)
     a_seq=np.random.permutation(a_seq.max() + 1)[a_seq]
     # axes[0].imshow(a_seq,cmap='spectral')
     
     # axes[1].axis('off')
     # axes[1].margins(0, 0)
     # axes[1].set_title('lower')
     # b_seq, _ , _ =relabel_sequential(b)
     # b_seq=np.random.permutation(b_seq.max() + 1)[b_seq]
     # axes[1].imshow(b_seq,cmap='spectral')
     plt.imshow(a_seq,cmap='spectral')
     click_markers =plt.ginput(n=-1, timeout=-1)
     return  click_markers, a_seq




# def replace_bad_slice_in_test(vol_data,set_name):
#     replace_slice={}
#     replace_slice['Set_A']={0:1,33:34,51:52,79:78,80:81,108:107,109:110,111:112}
#     replace_slice['Set_B']={15:14,16:17,44:43,45:46,77:78}
#     replace_slice['Set_C']={14:15,74:75,86:87}

#     r_set = replace_slice[set_name]
#     for idx,slice_d in enumerate(vol_data):
#         if idx in r_set:
#             vol_data[idx]=vol_data[r_set[idx]]
#     return vol_data



# def predict_hd5_to_img(file,im_dir, mode ='distance'):
#   h5=h5py.File(file)
#   print(h5.keys())
#   seg_dict ={}
#   # use only distance currently
#   for set_n in set_names:
#       h5_path=set_n+'_'+data_names[0]
#       data=h5[h5_path]
#       im_save_dir=im_dir+'_'+set_n
#       if not os.path.exists(im_save_dir):
#           os.mkdir(im_save_dir)
#       data = np.array(data)
#       data = replace_bad_slice(data,set_n)
#       seg_array = np.zeros_like(data)
#       for idx,slice_d in enumerate(data):
#           #save_slice_figure(im_save_dir, data_names[0],idx,slice_d,mode)
#           print('processing im {} in {}'.format(idx,set_n))
#           seg_array[idx]=watershed_seg(slice_d)

#       seg3D=connect_2d_slice(seg_array)
#       seg3D, _ , _ =relabel_sequential(seg3D)
#       seg_dict[set_n]=seg3D
#       save_seg3D_to_image(seg3D,im_save_dir, data_names[0])
#       #for i in range(10):
#       #   show2pairs_slice(i,seg_array,seg_3D)
#       #pdb.set_trace()
#   make_seg_submission(seg_dict)


def connect_2d_slice(data,seg_slices_array):
     #seg_connector = Simple_MaxCoverage_3DSegConnector()
     seg_connector =NN_slice_3DSegConnector()
     seg3d = seg_connector(data[0:50], seg_slices_array[0:50])
     return seg3d

def get_rawim_or_labels(task_set,subset_name,data_set):
    data_config ='conf/cremi_datasets.toml' \
                 if task_set == 'valid' else \
                 'conf/cremi_datasets_test.toml' 
    orig_dataset = slice_dataset(sub_dataset=subset_name,
                                 data_config =data_config)

    if data_set =='image':
        data = orig_dataset.get_data()
    elif data_set =='label':
        data =orig_dataset.get_label()
    return data

def save_image_from_orig_volume(im_dir, task_set,subset_name, data_set='image'):
    '''Input Params:
       orig_set : 'valid' or 'test'
       subset_name: 'Set_A','Set_B','Set_C'
       data_set:   'image', 'seg_label'
    '''
    # data_config ='conf/cremi_datasets.toml' \
    #              if task_set == 'valid' else \
    #              'conf/cremi_datasets_test.toml' 
    # orig_dataset = slice_dataset(sub_dataset=subset_name,
    #                            data_config =data_config)

    # if data_set =='image':
    #   data = orig_dataset.get_data()
    # elif data_set =='seg_label':
    #   data =orig_dataset.get_label()

    data=get_rawim_or_labels(task_set,subset_name,data_set)

    im_save_dir=im_dir+'_'+subset_name
    if not os.path.exists(im_save_dir):
            os.mkdir(im_save_dir)

    if data_set == 'image':
        save_volume_to_image_slice(data,im_save_dir,data_name=data_set)
    elif data_set == 'label':
        data, _ , _ =relabel_sequential(data)
        save_seg3D_to_image(data,im_save_dir, data_name='_GT_')


    #for idx, slice in enumerate(data):


def save_distance_as_ProbMap_to_h5(hd5_file,im_dir):
    for set_n in set_names:
        h5=h5py.File(hd5_file)
        print(h5.keys())
        print('save {}:{} to prob h5'.format(set_n,'distance'))
        h5_path=set_n+'_'+'distance'
        data=np.array(h5[h5_path])
        data=torch.from_numpy(data)
        data =torch.sigmoid(data)
        data =1-data.numpy()
        #hd5_save_dir=im_dir+'_'+set_n
        if not os.path.exists(im_dir):
           os.mkdir(im_dir)
        
        dst_lb_hd5_file =os.path.join(im_dir,set_n+'_Prob.h5')
        h5=h5py.File(dst_lb_hd5_file,'w')
        h5.create_dataset('Prob', data = data,chunks=True)
        h5.close()

def save_2DSeg_to_h5(src_hd5_file, dst_dir):
    for set_n in set_names:
        h5=h5py.File(src_hd5_file)
        print('save {}:{} to image'.format(set_n,'2D segmentation'))
        h5_path=set_n+'_'+'distance'
        distance=np.array(h5[h5_path])

        rep_bad_slice = True
        h5.close()
        distance = replace_bad_slice(distance,set_n) \
                              if rep_bad_slice \
                              else distance



        seg_vol = np.zeros_like(distance).astype(np.uint32)
        for idx, slice_d in enumerate(distance):
            print('segmenting slice {} of {}'.format(idx,set_n))
            seg_vol[idx] = watershed_seg(slice_d,threshold=0.1)
            seg_vol[idx]+=idx*1000

        seg_vol,_ , _ =relabel_sequential(seg_vol)
        seg_vol = seg_vol.astype(np.uint32)
        dst_hd5_file =os.path.join(dst_dir,set_n+'seg2D.h5')
        h5=h5py.File(dst_hd5_file,'w')
        h5.create_dataset('seg_2D', data = seg_vol, chunks=True,dtype='uint32')
        h5.close

def save_raw_and_seglabel_to_h5(task_name,dst_dir):
    for set_n in set_names:
        im,lb=read_raw_image_data(task_name,set_n)
        rep_bad_slice =True 
        if task_name == 'test':
            im = replace_bad_slice(im,set_n) \
                              if rep_bad_slice \
                              else im
        #pdb.set_trace()
        h5_im_path=set_n+'_'+'GT.h5'
        print('save label and image of {}'.format(set_n))
        dst_im_hd5_file =os.path.join(dst_dir,set_n+'_raw.h5')
        h5=h5py.File(dst_im_hd5_file,'w')
        h5.create_dataset('raw', data=im,chunks=True)
        h5.close()
        if task_name =='valid':
            dst_lb_hd5_file =os.path.join(dst_dir,set_n+'_GT.h5')
            h5=h5py.File(dst_lb_hd5_file,'w')
            h5.create_dataset('gt_label', data = lb,chunks=True)
            h5.close()


def save_all_distance_to_image_slice(hd5_file,im_dir):
    for set_n in set_names:
        h5=h5py.File(hd5_file)
        print(h5.keys())
        print('save {}:{} to image'.format(set_n,'distance'))
        h5_path=set_n+'_'+'distance'
        data=np.array(h5[h5_path])

        im_save_dir=im_dir+'_'+set_n
        if not os.path.exists(im_save_dir):
            os.mkdir(im_save_dir)
        save_volume_to_image_slice(data,im_save_dir,'distance')


def save_volume_to_image_slice(vol,im_save_dir,data_name):
    for idx,im_slice in enumerate(vol):
        fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
        plt.imshow(im_slice,cmap='gray')
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(os.path.join(im_save_dir,data_name+'_slice_'+str(idx)+'.png'))
        plt.close()
        print('saving vol slice: {}'.format(idx))

def save_seg3D_to_image(seg3D,im_save_dir, data_name):
    seg3D =np.random.permutation(seg3D.max() + 1)[seg3D]
    for idx,seg_slice in enumerate(seg3D):
        fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)
        plt.imshow(seg_slice,cmap='spectral')
        plt.axis('off')
        plt.margins(0, 0)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(os.path.join(im_save_dir,data_name+'_3D_seg_'+str(idx)+'.png'))

        #plt.show(block=False)
        
        #plt.close()
        print('saving 3d_connect im {}'.format(idx))







def show2pairs_slice(idx,seg_2d,seg_3d):
  fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
  seg2d_pair, _ , _ =relabel_sequential(seg_2d[idx:idx+2])
  seg2d_pair=np.random.permutation(seg2d_pair.max() + 1)[seg2d_pair]

  seg3d_pair, _ , _ =relabel_sequential(seg_3d[idx:idx+2])
  seg3d_pair=np.random.permutation(seg3d_pair.max() + 1)[seg3d_pair]
  

  axes[0, 0].imshow(seg3d_pair[0],cmap='spectral')
  axes[0, 0].axis('off')
  axes[0, 0].margins(0, 0)
  axes[0,0].set_title('seg3d-1')

  axes[0, 1].imshow(seg3d_pair[1],cmap='spectral')
  axes[0, 1].axis('off')
  axes[0, 1].margins(0, 0)
  axes[0, 1].set_title('seg3d-2')

  axes[1, 0].imshow(seg2d_pair[0],cmap='spectral')
  axes[1, 0].axis('off')
  axes[1, 0].margins(0, 0)
  axes[1,0].set_title('seg2d-1')

  axes[1, 1].imshow(seg2d_pair[1],cmap='spectral')
  axes[1, 1].axis('off')
  axes[1, 1].margins(0, 0)
  axes[1, 1].set_title('seg2d-2')
  plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
  #os.path.join(im_save_dir,data_name+suffix+'_3D_seg_'str(idx)+'.png')
  #plt.show()

def save_slice_figure(im_save_dir, data_name,idx,slice_d, mode):
    fig = plt.figure(figsize=(1250/my_dpi, 1250/my_dpi), dpi=my_dpi)

    if mode =='watershed':
        slice_d=watershed_seg(slice_d)
        plt.imshow(np.random.permutation(slice_d.max() + 1)[slice_d],
                                                         cmap='spectral')
        suffix ='_segment_'
    elif mode=='distance':
        plt.imshow(slice_d)
        suffix ='_distance_'
    else:
        print('mode : {} is not valid'.format(mode))
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(os.path.join(im_save_dir,data_name+suffix+str(idx)+'.png'))
    plt.close()


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





if __name__ == '__main__':
    set_to_process      = 'test'
    #task = 'save_dist_image'
    #task = 'save_gt_label'
    #task ='segmentation_submission'
    #task ='save_raw_image'
    #task ='guid'
    task ='Seg2D_h5'
    #task ='save_im_gt_h5'
    #task ='save_prob_h5'
    data_name_for_seg   = 'distance'

    h5_file_name_dict ={'valid':'tempdata/best_2D_distance_predict_validationSet.h5', \
                        'test':'tempdata/best_2D_distance_predict.h5'}
    
    image_dir_dict    ={'valid':'tempdata/valid_slice_image', \
                         'test':'tempdata/test_slice_image'}
    
    h5_file_name = h5_file_name_dict[set_to_process]
    image_dir    = image_dir_dict[set_to_process]


    #pdb.set_trace()
    if task =='save_dist_image':
        save_all_distance_to_image_slice(h5_file_name,image_dir)
    elif task =='save_prob_h5':
        save_distance_as_ProbMap_to_h5(h5_file_name,'tempdata')
    elif task =='segmentation_submission':
        segment_all_and_makeSubmission(h5_file_name,data_name_for_seg = 'distance',\
                                  im_dir=image_dir,\
                                  task_set=set_to_process,\
                                  save_seg_to_img=True, \
                                  rep_bad_slice=True)
    elif task =='save_raw_image':
        for subset in set_names:
            save_image_from_orig_volume(im_dir=image_dir,task_set=set_to_process, subset_name =subset,data_set='image')
    elif task == 'save_gt_label':
        for subset in set_names:
            save_image_from_orig_volume(im_dir=image_dir,task_set=set_to_process, subset_name =subset,data_set='label')
    elif task =='guid':
        guidedSegmentation(h5_file_name)
        #print('invalid task name: {}'.format(task))
    elif task =='Seg2D_h5':
        save_2DSeg_to_h5(h5_file_name,'tempdata')
    elif task == 'save_im_gt_h5':
        save_raw_and_seglabel_to_h5(set_to_process, 'tempdata')