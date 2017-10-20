import tensorflow as tf
import numpy as np
def conv2d(inputs, num_outputs, kernel_size, scope, norm=True,
           d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=d_format, activation_fn=None, biases_initializer=None)

    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
        data_format=d_format)


def conv3d(inputs,num_outputs,kernel_size):
	outputs = tf.contrib.layers.conv3d(
		inputs, num_outputs, kernel_size)
	 return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
        updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
        data_format=d_format)
