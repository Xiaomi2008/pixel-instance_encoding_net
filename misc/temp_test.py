import numpy as np
data =np.zeros([22,256,256])
seg_label=np.zeros([22,256,256])
# random crop each slize in xy plane, so that we can augmented mis-aligment.
# after crop it relies on the interprolatation to recover it size to fit the output size setting
mis_align_scale  =0.1
d_x_size = data.shape[2]
d_y_size = data.shape[1]
x_offset = int(d_x_size *mis_align_scale)
y_offset = int(d_y_size *mis_align_scale)
d_new_x_size = d_x_size - x_offset-1
d_new_y_size = d_y_size - y_offset-1
new_d_list =[]
new_seg_list=[]
for j in range(len(data)): # length of z-direction
    x_start = np.random.randint(0,x_offset)
    y_start = np.random.randint(0,y_offset)
    x_end   = x_start+d_new_x_size-1
    y_end   = y_start+d_new_y_size-1
    new_d_list.append(data[j, x_start:x_end, y_start:y_end])
    new_seg_list.append(seg_label[j, x_start:x_end, y_start:y_end])
    for i,nda in enumerate(new_d_list):
        print('s {} = {}'.format(i,nda.shape))
data = np.stack(new_d_list,0)
seg_label = np.stack(new_seg_list,0)