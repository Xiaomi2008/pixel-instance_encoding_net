
import torch
import tensorflow as tf
def l2_norm(x):
    return x/torch.sqrt(torch.max(x**2,1)[0])

def test_torch_acos_loss(gt,pred):
    pred        = pred.view(-1, outputChannels)
    gt          = gt.view(-1, outputChannels)
    pred        = l2_norm(pred)*0.999999
    gt          = l2_norm(gt)*0.999999
    p_xy  =pred[:,0]/torch.sqrt(torch.sum((pred*pred),1))   
    gt_xy =gt[:,0]/torch.sqrt(torch.sum((gt*gt),1))
    err_angle= torch.acos(p_xy) - torch.acos(gt_xy)
    loss = torch.sum(err_angle*err_angle)
    return loss
def test_tf_acros_loss(gt,pred):
    pred = tf.reshape(pred, (-1, outputChannels))
    gt = tf.to_float(tf.reshape(gt, (-1, outputChannels)))
    pred = tf.nn.l2_normalize(pred, 1) * 0.999999
    gt = tf.nn.l2_normalize(gt, 1) * 0.999999
    errorAngles = tf.acos(tf.reduce_sum(pred * gt, reduction_indices=[1], keep_dims=True))
    loss        = tf.reduce_sum(tf.abs(errorAngles*errorAngles))
    return loss
def main_test():
    sess = tf.Session()
    pred =[[2,3,100,9,100],[4,5,6,0.2,888]]
    gt   =[[0.1,0.8,1999,0.89,45],[1000,0.2,3333,0.22,787]]
    loss = test_tf_acros_loss(tf.constant(gt),tf,constant(pred))
    print(sess.run(loss))
if __name__ == "__main__":
    main_test()