import sys
#sys.path.append('../')
import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis, binary_erosion,binary_opening, \
                               binary_closing,binary_dilation, disk, \
                               skeletonize,skeletonize_3d
from scipy.ndimage.morphology import distance_transform_edt as dis_transform
import matplotlib.pyplot as plt
from skimage.color import label2rgb
#from skimage.morphology import skeletonize, skeletonize_3d

from utils.EMDataset import slice_dataset
from utils.transform import *
from skimage.segmentation import watershed
import skimage
import pdb
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi

def dataset(slice_axis):
            return slice_dataset(sub_dataset='All',
                                     subtract_mean=True,
                                     split='valid',
                                     slices=1,
                                     slice_axis = 0,
                                     data_config= 'conf/cremi_datasets.toml')


#data = microstructure(l=128)
#dset = dataset(slice_axis=0)
#dset.set_current_subDataset('Set_B')

#data_list = [np.squeeze(dset.__getitem__(i)['label'].cpu().numpy())[:1248,:1248] for i in range(80,125)]
#gt_seg =np.stack(data_list,0)


aff_x = affinity(axis = -1,distance =16)
aff_y = affinity(axis = -2,distance =16)
aff_z = affinity(axis = -3,distance =2)
#compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0] +aff_z(x)[0])==0).astype(np.int)
#compute_boundary = lambda x: ((aff_x(x)[0]+aff_y(x)[0])==0).astype(np.int)
#affinity = compute_boundary(gt_seg)


#gx,gy   =  np.gradient(dt,1,1,edge_order =1)


#print(data.shape)

# Compute the medial axis (skeleton) and the distance transform
#skel, distance = medial_axis(affinity, return_distance=True)
#
#
#
#
compute_per_slice = lambda x, func: np.stack([func(x[i]) for i in range(len(x))],0)

compute_per_slice2 = lambda x,y, func: np.stack([func(x[i],y[i]) for i in range(len(x))],0)

def get_gt_labels(set_name='Set_C', start_slice=80, end_slice=125):
    dset = dataset(slice_axis=0)
    dset.set_current_subDataset(set_name)
    data_list = [np.squeeze(dset.__getitem__(i)['label'].cpu().numpy())[:1250,:1250] for i in range(start_slice,end_slice)]
    gt_seg =np.stack(data_list,0)
    return gt_seg
    


def watershed3D(seg_gt, threshold):
    b1 = ((aff_x(seg_gt)[0] + aff_y(seg_gt)[0])==0).astype(np.int)
    b2 =(aff_z(seg_gt)[0]==0).astype(np.int)
    #markers = skimage.morphology.label(b1)
    #distance = compute_per_slice(b1,dis_transform)
    #
    #distance = compute_per_slice(b1,dis_transform)
    distance = dis_transform((b1 &b2).astype(np.int))
    threds = np.linspace(0.1, 15, num=25)
    for th in threds:
        markers = distance > th
        markers = skimage.morphology.label(markers)
        print('threshold = {}'.format(th))
        #markers = skimage.morphology.label(b1&b2)
        #plt.imshow(markers[1])
        #plt.show()
        seg_pred =watershed(-distance,markers)
        evaluate(seg_gt, seg_pred)
    return seg_pred

def watershed2D_and_affinZ_connect_method(gt_seg):
    b1 = ((aff_x(gt_seg)[0] + aff_y(gt_seg)[0])==0).astype(np.int)
    b2 =(aff_z(gt_seg)[0]==0).astype(np.int)
    #markers = skimage.morphology.label(b1)
    distance = compute_per_slice(b1,dis_transform)
    makers   = compute_per_slice(b1,skimage.morphology.label)
    seg_2D   = compute_per_slice2(-distance,makers, watershed)
    seg_2D   = np.stack([seg_2D[i] + (i*1000) for i in range(len(seg_2D))],0)
    print(len(seg_2D))
    # for i in range(len(seg_2D)):
    #     seg_2D[i][seg_2D[i] ==(i*1000)] =0

    #A = 
    
    


    #seg_2D   = np.stack([seg_2D[i][seg_2D[i] ==(i*2000)]=0 for i in range(len(seg_2D))],0)
    #seg_2D[seg_2D == (i*2000)] =0
    #
    #if i ==0:
    #plt.imshow(seg_2D[1]==1118)
    #plt.show()

    for i in range(0,len(seg_2D)-1):
        ids_pair=get_next_slice_connected_ids(seg_2D,b1,b2,i)
        #print(ids_pair)
        #pdb.set_trace()
        seg_2D=connect(seg_2D,i,ids_pair)
        #plot_ims(seg_2D[i],seg_2D[i+1], gt_seg[i], gt_seg[i+1])
        #pdb.set_trace()


    return seg_2D

def connect(seg, slice_idx, ids_pair):
    connected_ids ={}
    for id,ids_next in ids_pair.iteritems():

        print('id {} = ids {}'.format(id,ids_next))
        if id ==0:
            continue
        for id_next in ids_next:
            if id_next  not in connected_ids:
                seg[slice_idx+1][seg[slice_idx+1]==id_next] = id
                connected_ids[id_next] =id
            else:
                pass
                #seg[seg== connected_ids[id_next]] = id
                # for item in ids_pair[connected_ids[id_next]]:
                #     if item in connected_ids:
                #         connected_ids[item] = id
                        

                        #print('connected')
                #seg[ids+1][seg[ids+1] == id_next] = id
                #connected_ids.add()
    return seg

def get_next_slice_connected_ids(seg,affine_xy, affine_z,slice_idx):

    ids = np.unique(seg[slice_idx])
    conn_ids_pair={}
    for id in ids:
        #nextslice_ids=seg[slice_idx+1][seg[slice_idx] == id and b2[slice_idx] ==1]
        n_seg = seg[slice_idx+1]
        A = seg[slice_idx] == id
        B = affine_z[slice_idx] ==1
        C = affine_xy[slice_idx] ==1
        #pdb.set_trace()
        matched_ids = n_seg[A&B&C]
        unique_ids = np.unique(matched_ids)
        #unique_ids = list(filter(lambda x: x >0, np.unique(matched_ids)))

        # if id  == 272:
        #     C = (n_seg==unique_ids[0]).astype(np.int)
        #     pdb.set_trace()
        #     plot_ims(A,B,C,B)
        conn_ids_pair[id] = unique_ids
    return conn_ids_pair


def evaluate(seg_gt,seg_pred):
    arand = adapted_rand(seg_pred.astype(np.int), seg_gt)
    split, merge = voi(seg_pred.astype(np.int), seg_gt)
    print('rand , voi (Merg, Split) ={:.3f}, ({:.3f},{:.3f})'.format(arand,merge,split))

def plot_ims(p_seg1,p_seg2,g_seg1,g_seg2):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0,0].imshow(p_seg1, cmap=plt.cm.spectral, interpolation='nearest')
    axs[0,0].title.set_text('p_seg_1')
    axs[0,0].axis('off')

    axs[0,1].imshow(p_seg2, cmap=plt.cm.spectral, interpolation='nearest')
    axs[0,1].title.set_text('p_seg_2')
    axs[0,1].axis('off')



    axs[1,0].imshow(g_seg1)
    axs[1,0].title.set_text('g_seg_1')
    axs[1,0].axis('off')

    axs[1,1].imshow(g_seg2, cmap=plt.cm.spectral, interpolation='nearest')
    axs[1,1].title.set_text('g_seg_2')
    axs[1,1].axis('off')

    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()




if __name__ == '__main__':
    seg_gt = get_gt_labels(set_name='Set_C', start_slice=80, end_slice=105)
    #seg_pred = watershed2D_and_affinZ_connect_method(seg_gt)
    seg_pred = watershed3D(seg_gt,10)
    evaluate(seg_gt, seg_pred)
    i=0
    plot_ims(seg_pred[i],seg_pred[i+1], seg_gt[i], seg_gt[i+1])
    

#distance =  np.stack([dis_transform(affinity[i]) for i in range(len(affinity))],0)

#distance = dis_transform(affinity)

#skel =skeletonize_3d(affinity)

# skel_mask=ndimage.gaussian_filter(skel, sigma=l/(4.*n))
# skel = (skel > 0.4).astype(int)

#structure = disk(2, dtype=bool)
#skel = binary_dilation(binary_dilation(skel))
#skel = binary_dilation(skel,structure)
# dist_on_skel = distance #* skel
# #markers = skimage.morphology.label((dist_on_skel>1.5).astype(int))
# #markers = skimage.morphology.label(skel)

# gt_seg =gt_seg[1:2]
# distance = distance[1:2]
# affinity =affinity[1:2]
# seg2 = (1-affinity) * gt_seg

# markers = skimage.morphology.label(affinity)
# seg_labels = watershed(-distance, markers)


# print('dt = {}'.format(distance.shape))
# #gz,gx,gy =  np.gradient(distance,1,1,1, edge_order =1)
# gx,gy =  np.gradient(distance[0],1,1, edge_order =1)

# # Distance to the background for pixels of the skeleton
# #dist_on_skel = distance * skel12

# #dist_on_skel = binary_erosion(skel)
# #
# #
# #
# #
# #seg_labels[seg_labels ==28]=79
# #seg_labels[seg_labels ==93]=79

# arand = adapted_rand(seg_labels.astype(np.int), gt_seg)
# split, merge = voi(seg_labels.astype(np.int), gt_seg)



# print('rand , voi Merg, Split ={:.3f}, ({:.3f},{:.3f})'.format(arand,merge,split))
# slice_idx =0

# fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# axs[0,0].imshow(seg_labels[slice_idx], cmap=plt.cm.spectral, interpolation='nearest')
# axs[0,0].title.set_text('seg_waterseg')
# axs[0,0].axis('off')

# axs[0,1].imshow(gt_seg[slice_idx], cmap=plt.cm.spectral, interpolation='nearest')
# #axs[0,1].contour(data[slice_idx], [0.5], colors='w')
# axs[0,1].title.set_text('gt')
# axs[0,1].axis('off')

# axs[1,0].imshow(gy)
# axs[1,0].title.set_text('gradient')
# axs[1,0].axis('off')

# axs[1,1].imshow(distance[slice_idx])
# axs[1,1].title.set_text('distance')
# axs[1,1].axis('off')

# fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
# plt.show()