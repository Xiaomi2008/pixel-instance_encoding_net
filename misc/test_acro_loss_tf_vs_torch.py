
import torch
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import pdb
outputChannels =2 
def tc_l2_norm(x):
    #print x
    epsilon=torch.FloatTensor([1e-12])
    sq_x = torch.max(x**2,epsilon)
    sum_x = torch.sum(sq_x,0)
    sqrt_x = torch.sqrt(sum_x)
    #return sum_x
    # print('sqrt_x  = {}'.format(sqrt_x))
    return x/sqrt_x
def tf_l2_norm(x,dim):
    epsilon=1e-12
    name='l2norm'
    x = ops.convert_to_tensor(x, name="x")
    sq= math_ops.square(x)
    square_sum = math_ops.reduce_sum(sq, dim, keep_dims=True)
    #return square_sum
    x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
    return math_ops.multiply(x, x_inv_norm, name=name)

def test_torch_acos_loss(gt,pred):
    #pred        = pred.view(-1, outputChannels)
    #gt          = gt.view(-1, outputChannels)
    #pred        = tc_l2_norm(pred)*0.999999
    #gt          = tc_l2_norm(gt)*0.999999
    print('gt = {}'.format(gt))
    p_xy        = pred[0,:]/torch.sqrt(torch.sum((pred*pred),0))   
    gt_xy       = gt[0,:]/torch.sqrt(torch.sum((gt*gt),0))
    err_angle   = torch.acos(p_xy) - torch.acos(gt_xy)
    loss        = torch.sum(err_angle*err_angle)
    return loss
def test_torch_acos_loss2(gt,pred):
    #pred        = pred.view(-1, outputChannels)
    #gt          = gt.view(-1, outputChannels)
    pred        = tc_l2_norm(pred)*0.999999
    gt          = tc_l2_norm(gt)*0.999999
    prod_sum    = torch.sum(gt*pred,0)
    #print prod_sum
    angle_err   = torch.acos(prod_sum)
    loss        = torch.sum(angle_err*angle_err)
    return loss
def test_tf_acos_loss(gt,pred):
    pred  = tf.transpose(pred,perm=[1,0])
    pred  = tf.reshape(pred, (-1, outputChannels))

    gt    = tf.transpose(gt,perm=[1,0])
    gt    = tf.to_float(tf.reshape(gt, (-1, outputChannels)))
    
    pred = tf.nn.l2_normalize(pred, 1) * 0.999999
    gt = tf.nn.l2_normalize(gt, 1) * 0.999999

    xr_ratio =tf.reduce_sum(pred * gt, reduction_indices=[1], keep_dims=True)


    #print(xr_ratio)
    
    errorAngles = tf.acos(xr_ratio)
    sum_err_sqrt = errorAngles*errorAngles
    loss        = tf.reduce_sum(tf.abs(sum_err_sqrt))
    return loss

def test_torch_l2_norm(gt,pred):
    #print pred
    # pred      = pred.view(-1, outputChannels).contiguous()
    # gt        = gt.view(-1, outputChannels).contiguous()
    pred      = tc_l2_norm(pred)*0.999999
    #pred      = pred.numpy()
    #gt          = l2_norm(gt)*0.999999

    return pred

def test_tf_l2_norm(gt,pred):
    pred  = tf.transpose(pred,perm=[1,0])
    pred = tf.reshape(pred, (-1, outputChannels))
    #print (pred)
    #gt = tf.to_float(tf.reshape(gt, (-1, outputChannels)))
    pred = tf_l2_norm(pred, 1) * 0.999999
    # gt   = tf_l2_norm(gt, 1) * 0.999999
    # # pred = tf.nn.l2_normalize(pred, 1) * 0.999999
    # # gt = tf.nn.l2_normalize(gt, 1) * 0.999999

    return pred

def main_test():
    sess = tf.Session()
    pred =[[2,3,100,9,100],[4,5,6,0.2,888]]
    gt = pred
    #gt   =[[0.1,0.8,1999,0.89,45],[1000,0.2,3333,0.22,787]]


    tf_pred, tf_gt = tf.constant(pred), tf.constant(gt)
    
    tc_pred, tc_gt = torch.FloatTensor(pred),torch.FloatTensor(gt)
    #tf_l2 = test_tf_l2_norm(tf.constant(gt),tf.constant(pred))
    #tf_sum = tf.reduce_sum(tf_l2,reduction_indices = [1])

    tf_acos_loss = test_tf_acos_loss(tf_gt,tf_pred)

    print(sess.run(tf_acos_loss))
    print(test_torch_acos_loss(tc_gt,tc_pred))
    #print(sess.run(tf_l2))
    #print(test_torch_l2_norm(torch.FloatTensor(gt),torch.FloatTensor(pred)))
    # loss = test_tf_acros_loss(tf.constant(gt),tf.constant(pred))
    # print(sess.run(loss))

    # print(test_torch_acos_loss(torch.FloatTensor(gt),torch.FloatTensor(pred)))


    # print(test_torch_acos_loss2(torch.FloatTensor(gt),torch.FloatTensor(pred)))
if __name__ == "__main__":
    main_test()