# : Harvey Chang
# : chnme40cs@gmail.com
# this file is used to define a wavelet function for activation:
import tensorflow as tf
import numpy as np


def gauss_activation(x):
    return (1 - 2 * tf.pow(x, 2)) * tf.exp(x)


def wavelet(input_tensor, activation=gauss_activation):
    # wnn only accept specific shape
    # for example [10*120: 10 group and 120 channels]->[10, 12]: 10->1
    shape = input_tensor.shape
    # reshape into one array:
    input_channel = shape[2].value
    print('input channel is {}'.format(input_channel))
    # add a W
    # first is the random initializer
    W = tf.Variable(tf.random_normal(shape=[1, 1, input_channel], stddev=0.01))
    # square W
    sqW = tf.sqrt(tf.abs(W))
    b = tf.Variable(tf.zeros([1, 1, input_channel]))
    # broadcasting multiply:
    mid = input_tensor * W + b
    mid = gauss_activation(mid)
    # divide sqW
    out = mid*sqW
    assert out.shape == shape
    eval_value = mid
    return out, W


# test the layer:
if __name__ == '__main__':
    input_tensor = tf.zeros([1024, 10, 10])
    wnn_tensor, eval_value = wavelet(input_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        e = sess.run(eval_value)
    print(e)