import numpy as np
from matplotlib import pyplot as plt
from skimage.color import label2rgb
import h5py
from utils.evaluation import adapted_rand
from misc.orig_cremi_evaluation import voi
import torch
from utils.utils import watershed_seg, watershed_seg2
from torch_networks.resnext import resnext50
from torch.autograd import Variable
from utils.mask_slice_pair_dataset import crop_and_resize
from utils.EMDataset import slice_dataset
import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation
from skimage.measure import label
import pdb
import sys
test_z_len =123
# def get_data(hf5,set_name = 'Set_A'):
#   # d_orig = hf5['d1_'+set_name]
#   # d_combine =hf5['d2_'+set_name]
#   # tg        =hf5['t1_'+set_name]
#   d_orig = hf5[set_name +'_d1']
#   d_combine =hf5[set_name +'_d2']
#   tg        =hf5[set_name +'_t1']
#   return d_orig, d_combine, tg

def build_network(weight_file):
    net_model=resnext50(8, 5, 2)
    print('load network weight: {}'.format(weight_file))
    net_model.load_state_dict(torch.load(weight_file))
    if torch.cuda.is_available():
        print('use GPU model')
        net_model = net_model.cuda()

    net_model.eval()
    return net_model


def build_2Dslice_ids_from_3Dseg(gt_label):
    z_slice=len(gt_label)
    gt_slice_diff_lable=gt_label.copy()
    # re-label each slice
    gt_slice_diff_lable=np.stack([label(gslice)+1000*idx for idx, gslice in enumerate(gt_slice_diff_lable)],axis =0)

    return gt_slice_diff_lable


def compute_adj_matrix(net_model, image, gt_label,use_true_label =False):
    z_slice=len(gt_label)
    
    gt_slice_diff_lable = build_2Dslice_ids_from_3Dseg(gt_label)
    adj_matrix, slice_ids_and_counts, (flat_obj_ids_list, flat_obj_ids_counts) \
                        = build_adjecency_objMatrix(gt_slice_diff_lable)
    
    ids_idx_dict = {sid:idx for idx,sid in enumerate(flat_obj_ids_list)}

    #test_z_len=10
    for slice_idx in range(0,test_z_len): #we assume that first axis is z-axis
        gt_slice = gt_slice_diff_lable[slice_idx]
        #gt_slice = gt_label[slice_idx]
        im_input = image[slice_idx:slice_idx+2].copy() -127.0
        for idx,(s_id,count) in enumerate(slice_ids_and_counts[slice_idx]):
            mask = (gt_slice == s_id).astype(np.int)
            if use_true_label:
                 gt_id = np.unique(gt_label[slice_idx][mask.astype(np.bool)])
                 assert len(gt_id)==1
                 gt_id=gt_id[0]
                 #print('use true label')
                 g_slice = gt_label[slice_idx+1]
                 g_next = gt_slice_diff_lable[slice_idx+1]
                 probs,ids=compute_groundTruth_slice_connect_probs(s_id,
                                                                    mask,
                                                                    g_next,
                                                                    gt_id,
                                                                    g_slice)
               
            else:
                 probs,ids=compute_next_masked_slice_connect_probs(net_model,
                                                                   mask,
                                                                   s_id,
                                                                   im_input, 
                                                                   gt_slice_diff_lable[slice_idx+1])
            if isinstance(probs, Variable):
                probs = probs.data.cpu().numpy()
            for prob,next_id in zip(probs[:,1],ids):
                #if prob >0.7:
                adj_matrix[ids_idx_dict[s_id],ids_idx_dict[next_id]] =prob
                adj_matrix[ids_idx_dict[next_id],ids_idx_dict[s_id]] =prob
                    
        print('slice {} of {}'.format(slice_idx,z_slice-1) )
    return adj_matrix,gt_slice_diff_lable,flat_obj_ids_list

def test_slice_connector_on_GTdata(net_model, image, gt_label):
    z_slice=len(gt_label)
    # gt_slice_diff_lable=gt_label.copy()
    # # re-label each slice
    # #d=label(gt_slice_diff_lable[1])

    # gt_slice_diff_lable=np.stack([label(gslice)+1000*idx for idx, gslice in enumerate(gt_slice_diff_lable)],axis =0)
    # #pdb.set_trace()
    #for slice_idx in range(z_slice-1): #we assume that first axis is z-axis

    #gt_slice_diff_lable = build_2Dslice_ids_from_3Dseg(gt_label)


    gt_slice_diff_lable = build_2Dslice_ids_from_3Dseg(gt_label)
    adj_matrix, slice_ids_and_counts, (flat_obj_ids_list, flat_obj_ids_counts) \
                        = build_adjecency_objMatrix(gt_slice_diff_lable)
    
    ids_idx_dict = {sid:idx for idx,sid in enumerate(flat_obj_ids_list)}

    test_z_len=10
    for slice_idx in range(0,test_z_len): #we assume that first axis is z-axis
        gt_slice = gt_slice_diff_lable[slice_idx]
        #pdb.set_trace()
        #lb_ids,counts = slice_ids_and_counts[slice_idx]
        
        #lb_ids,counts = np.unique(gt_slice,return_counts=True)
        # sort_idx = np.argsort(counts)[::-1] # we want the descent order
        # lb_ids   = lb_ids[np.argsort(sort_idx)]
        # counts   = counts[np.argsort(sort_idx)]
        im_input = image[slice_idx:slice_idx+2].copy() -127.0
        #pdb.set_trace()
        gt_connected=set()
        #for idx,(s_id,count) in enumerate(zip(lb_ids,counts)):
        for idx,(s_id,count) in enumerate(slice_ids_and_counts[slice_idx]):
            mask = (gt_slice == s_id).astype(np.int)
            probs,ids=compute_next_masked_slice_connect_probs(net_model,
                                                              mask,
                                                              s_id,
                                                              im_input, 
                                                              gt_slice_diff_lable[slice_idx+1])

            

            for prob,next_id in zip(probs.data.cpu().numpy()[:,1],ids):
                #if prob >0.7:
                adj_matrix[ids_idx_dict[s_id],ids_idx_dict[next_id]] =prob
                adj_matrix[ids_idx_dict[next_id],ids_idx_dict[s_id]] =prob
                    # if next_id not in gt_connected:
                    #     fill_mask =  (gt_slice_diff_lable[slice_idx+1] ==next_id)
                    #     gt_slice_diff_lable[slice_idx+1,:,:][fill_mask] = s_id
                    #     gt_connected.add(next_id)
                    # else:
                    #     fill_mask =  (gt_slice_diff_lable==s_id)

            #print('slice = {}, id ={}, probs ={}'.format(slice_idx,s_id,probs))
            #s_ids_in_next_slice = np.unique(gt_label[slice_idx +1][mask])
        print('sclie {} of {}'.format(slice_idx,z_slice-1) )


    connected_3D_seglabel=adj_matrix_to_3D_segLabel(adj_matrix,gt_slice_diff_lable,flat_obj_ids_list)


    #pdb.set_trace()
    arand = adapted_rand(connected_3D_seglabel.astype(np.int)[0:test_z_len], gt_label[0:test_z_len])
    split, merge = voi(connected_3D_seglabel.astype(np.int)[0:test_z_len], gt_label[0:test_z_len])

    print('arand : {} (split, merge) : ({},{})'.format(arand,split,merge))

def get_data(set_name='Set_C'):
    dataset = slice_dataset(sub_dataset='All',
                            subtract_mean=True,
                            split='valid')
    dataset.set_current_subDataset(set_name)
    data=dataset.get_data()
    label=dataset.get_label()


    return data, label

struct         = ndimage.generate_binary_structure(2, 3)
struct         = ndimage.iterate_structure(struct, 2).astype(int)

def compute_groundTruth_slice_connect_probs(mask_id,mask,label_in_next_slice, gt_id,gt_next_slice):
    # struct         = ndimage.generate_binary_structure(2, 3)
    # struct         = ndimage.iterate_structure(struct, 2).astype(int)
    #dilate_mask    = ndimage.binary_dilation(mask, structure=struct).astype(np.bool)
    

    #sid_list = np.unique(label_in_next_slice[dilate_mask]).tolist()

    sid_list = np.unique(label_in_next_slice[mask.astype(np.bool)]).tolist()

    gt_sid_list =[]
    for sid in sid_list:
        sid_mask = (label_in_next_slice == sid)
        gt_id=np.unique(gt_next_slice[sid_mask])
        assert len(gt_id)==1
        gt_sid_list.append(gt_id[0])




    #gt_sid_list = np.unique(gt_next_slice[mask.astype(np.bool)]).tolist()
    
    connected = [gt_id == gt_mask_id for gt_mask_id in gt_sid_list]
    #pdb.set_trace()
    out = np.zeros([len(connected),2]).astype(np.float)
    out[:,1] = np.squeeze(np.array(connected).astype(np.float))
    #print('compute one slice')

    return out, sid_list

def compute_next_masked_slice_connect_probs(net_model,mask, mask_id,image, label_in_next_slice):
    #need to change mask data type back to bool for faster indexing #
    #struct         = ndimage.generate_binary_structure(2, 3)
    #struct         = ndimage.iterate_structure(struct, 2).astype(int)
    dilate_mask    = ndimage.binary_dilation(mask, structure=struct).astype(np.bool)
    s_ids_slice = np.unique(label_in_next_slice[dilate_mask.astype(np.bool)])
    #s_ids_slice = np.unique(label_in_next_slice[mask.astype(np.bool)])
    #print(s_ids_slice)
    #input_image =np.stack([image,next_slice_image],axis=0)
    sid_list =[]
    input_list=[]
    for idx,sid in enumerate(s_ids_slice):
        next_mask = (label_in_next_slice == sid).astype(np.int)
        input_mask =np.stack([mask,next_mask],axis =0)
        #pdb.set_trace()
        input_im,input_msk=crop_and_resize(image, input_mask, out_size=[224,224])
        net_input = np.concatenate([input_im,input_msk],axis=0)
        input_list.append(net_input)
        sid_list.append(sid)

    input_data = np.stack(input_list,axis=0)

    #pdb.set_trace()

    input_data=Variable(torch.from_numpy(input_data), volatile=True).float()
    if torch.cuda.is_available():
        input_data =input_data.cuda()

    #out = net_model(input_data)
    preds=net_model(input_data)
    out=torch.nn.Softmax()(preds)

    _, predicted = torch.max(preds.data, 1)
    predicted =predicted.cpu()

    prob = out.data.cpu().numpy()


    in_view_data =input_data.data.cpu().numpy()

    connected = [sid == mask_id for sid in sid_list]

    #pdb.set_trace()

    # if np.count_nonzero(prob[:,1] > 0.5) >1:
    #   for idx,img in enumerate(in_view_data):
    #       if prob[idx][1]>0.5:
    #           visualize_mask_image(img[2:],img[:2],prob[idx][1],idx,connected[idx])
    
    # if np.count_nonzero(prob[:,1] > 0.5) ==0:
    #     for idx,img in enumerate(in_view_data):
    #             visualize_mask_image(img[2:],img[:2],prob[idx][1],idx,connected[idx])

    #connect_probs = out[:,1]

    #return connect_probs, sid_list
    return out, sid_list

def visualize_mask_image(seg_mask,im,lb ,idx,connected):
  fig, axes = plt.subplots(nrows=2, ncols=2, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
  axes[0, 0].imshow(im[0], cmap='gray')
  axes[0, 0].axis('off')
  axes[0, 0].margins(0, 0)
  title_str = 'prob : {}, gt : {}'.format(lb,connected)
  axes[0,0].set_title(title_str)

  axes[0, 1].imshow(im[1], cmap='gray')
  axes[0, 1].axis('off')
  axes[0, 1].margins(0, 0)

  axes[1, 0].imshow(seg_mask[0])
  axes[1, 0].axis('off')
  axes[1, 0].margins(0, 0)

  axes[1, 1].imshow(seg_mask[1])
  axes[1, 1].axis('off')
  axes[1, 1].margins(0, 0)
  plt.savefig('mask_test{}.png'.format(idx))
  plt.show()


def build_adjecency_objMatrix(seg_valume_2Dlabel,sort_ids_by_size=True):
    num_slice = len(seg_valume_2Dlabel)
    slice_ids_and_counts =[]
    flatten_slice_ids=[]
    flatten_slice_id_count = []
    for slice_seg_label in seg_valume_2Dlabel:
        slice_ids,counts =np.unique(slice_seg_label, return_counts=True)
        if sort_ids_by_size:
           sort_idx   = np.argsort(counts)[::-1] # we want the descent order
           slice_ids  = slice_ids[np.argsort(sort_idx)]
           counts     = counts[np.argsort(sort_idx)]

        slice_ids_and_counts.append(zip(slice_ids,counts))
        flatten_slice_ids+=slice_ids.tolist()
        flatten_slice_id_count+=counts.tolist()
    total_num_slice_ids =len(flatten_slice_ids)

    adj_matrix=np.zeros([total_num_slice_ids,total_num_slice_ids])

    return adj_matrix, slice_ids_and_counts, (flatten_slice_ids,  flatten_slice_id_count)

def adj_matrix_to_3D_segLabel(adj_matrix, 
                             slice_gt_label,
                             flatten_slice_ids, 
                             threshold=0.6,
                             singleConnect = False):
    '''
    Convert the adjecentcy matrix of objects ids to 3D segementaions 
    '''


    def make_adj_matric_to_graph_dict(adj_matrix,flatten_slice_ids):
        flat_ids_arr =np.array(flatten_slice_ids)
        graph_dict = {}
        for idx,vec in enumerate(adj_matrix):
            obj_id = flat_ids_arr[idx]
            if not singleConnect:
                conn_nodes =  flat_ids_arr[(vec>threshold).astype(np.bool)].tolist()
            else:
                conn_nodes =  []  \
                              if max(vec)<=threshold else \
                              flat_ids_arr[vec==max(vec)].tolist()

            #pdb.set_trace()
            #conn_nodes = connected_ids_in_node.tolist()
            if len(conn_nodes):
                graph_dict[obj_id]=conn_nodes
        return graph_dict


    def bsf_connected_component(graph,start_node):
        explored =[]
        queue = [start_node]

        level ={}
        level[start_node]=0
        visited =[start_node]
        assert (start_node in graph)
        while queue:
            node =queue.pop(0)
            explored.append(node)
            if node in graph:
                neighbours =graph[node]
            else:
                neighbours=[]

            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)

                    level[neighbour] =level[node]+1
        #print(level)
        return explored

    

    #ids_queque = [graph_dict.keys()[0]]

    '''Find connected obj ids in the adjecency matrix '''
    graph_dict = make_adj_matric_to_graph_dict(adj_matrix,flatten_slice_ids)
    conned_node_list =[]
    while len(graph_dict):
        key =list(graph_dict.keys())[0]
        connected_nodes=bsf_connected_component(graph_dict,key)
        conned_node_list.append(connected_nodes)
        for node in connected_nodes:
            if node in graph_dict:
                del graph_dict[node]
        

        #print(graph_dict)

    print(len(conned_node_list))

    #pdb.set_trace()

    '''Assing id to connected voxels in 3D volume '''
    assign_id = 1
    new_3D_seg_label=np.zeros_like(slice_gt_label)


    # idexing and assinging id to a 3D array (segmentation )
    # in GPU matrix is much faster
    import torch
    
    new_3D_seg_label=torch.from_numpy(new_3D_seg_label).cuda()
    slice_gt_label=torch.from_numpy(slice_gt_label).cuda()

    for idx,conn_nodes in enumerate(conned_node_list):
        #mask_vol=np.zeros_like(slice_gt_label).astype(bool)
        if idx % int(len(conned_node_list)/10.0) == int(len(conned_node_list)/10.0):
            print('fill connected nodes {} of {}'.format(idx,len(conned_node_list)))
        #sys.stdout.flush()
        for node in conn_nodes:
            #mask_vol |= (slice_gt_label==node).astype(np.bool)
            new_3D_seg_label[slice_gt_label==node] =assign_id

        #new_3D_seg_label[mask_vol] = assign_id

        assign_id +=1        

    new_3D_seg_label =new_3D_seg_label.cpu().numpy()
    return new_3D_seg_label






    # for key in graph_dict.keys():
    #     connected_nodes=bsf_connected_component(graph_dict,key)
    #     print('{} connected node = {}'.format(key,connected_nodes))

    #new_3D_seg_label=np.zeros_like(slice_gt_label)



if __name__ == '__main__':


    load_from_file =False
    
    if load_from_file:
        import cPickle
        f =open('tempdata/adjmatrix_setB.pkl','rb')
        my_adj_dict=cPickle.load(f)
        gt_slice_diff_lable = my_adj_dict['seg2D']
        flat_obj_ids_list=my_adj_dict['flat_id_list']
        adj_matrix_prob=my_adj_dict['adjmatprob']
        set_name = 'Set_B'
        data, seg_label=get_data(set_name)
    else:
        #hf5 = h5py.File('tempdata/seg_final_plus_distance.h5','r')
        model_file = 'model/GT_Resnet50_b8_c5_Dataset-CRIME-All_mask_crossEntropy_loss_VFlip-HFlip-Rot90-NRot90_iter_181499.model'
        #hf5 = h5py.File('tempdata/seg_fina_distance_only.h5','r')
        #hf5 =h5py.File('tempdata/seg_mu1_distance.h5', 'r')
        #dataset = 'Set_A'
        #d_orig,d_combine,tg = get_data(hf5,dataset)
        #t    = tg[100:,:,:]

        '''get adj_matrix'''
        set_name = 'Set_A'
        data, seg_label=get_data(set_name)
        net_model = build_network(model_file)
        adj_matrix_prob,gt_slice_diff_lable,flat_obj_ids_list= compute_adj_matrix(net_model, data, seg_label,use_true_label=False)
        #test_slice_connector_on_GTdata(net_model, data, seg_label)

    

    '''connecting'''
    
    #connected_3D_seglabel=adj_matrix_to_3D_segLabel((adj_matrix_prob>prob_threshold).astype(np.int),gt_slice_diff_lable,flat_obj_ids_list)
    prob_threshold =0.8
    connected_3D_seglabel=adj_matrix_to_3D_segLabel(adj_matrix_prob,
                                                   gt_slice_diff_lable,
                                                   flat_obj_ids_list,
                                                   threshold=prob_threshold)

    arand = adapted_rand(connected_3D_seglabel.astype(np.int)[0:test_z_len], seg_label[0:test_z_len])
    split, merge = voi(connected_3D_seglabel.astype(np.int)[0:test_z_len], seg_label[0:test_z_len])

    print('prob_threshodl : {} for {},  arand : {} (split, merge) : ({},{})'.format(prob_threshold, set_name, arand,split,merge))

    # gt_slice_diff_lable = build_2Dslice_ids_from_3Dseg(seg_label)
    # adj_matrix, slice_ids_and_count, (obj_ids_list,obj_ids_counts) \
    #                     = build_adjecency_objMatrix(gt_slice_diff_lable)
    # ids_idx_dict = {sid:idx for idx,sid in enumerate(obj_ids_list)}



    # thresholds = np.linspace(16,35,15)
    # arands = []
    # print ('test {}'.format(dataset))
    # for th in thresholds:
    #   #d_seg= watershed_seg2(d_orig[100:,:,:], d_combine[100:,:,:], threshold = th)
    #   d_seg= watershed_seg(d_combine[100:,:,:], threshold = th)
    #   #d_seg= watershed_seg(d_orig[100:,:,:], threshold = th)
    #   arand = adapted_rand(d_seg.astype(np.int), t)
    #   split, merge = voi(d_seg.astype(np.int), t)
    #   arands.append(arand)
    #   print('arand, split, merge = {:.3f}, {:.3f}, {:.3f} for threshold = {:.3f}'.format(arand,split,merge,th))
    #   #print('arand ={}  for threshold= {}'.format(arand,th))
    # plt.plot(arands)
    # plt.title('Set_' + dataset)
    # plt.show()



