import tensorflow as tf
import numpy as np
import time
import os
import sys
from tensorflow.python.framework import ops
from src.Utils.nnUtils import msra_init, zeros_init, ones_init, binary


class nnUtils:

    def __init__(self):
        self.is_train = tf.placeholder(dtype = tf.bool)
        self.tensor_updated = []

    def bnn_conv2d(self, input_tensor, kernel, stride, init_func = msra_init, bn_input = True, padding = "SAME"):
        assert(len(kernel) == 4)
        assert(len(stride) == 2)
        W = init_func(kernel, "conv_weights")
        W = binary(W)
        if bn_input == True:
            input_tensor = binary(input_tensor)
        conv_res = tf.nn.conv2d(input_tensor, W, [1, stride[0], stride[1], 1], padding = padding)
        return conv_res
        
    
    def bn(self, input_tensor, scale, bias, is_conv = True):
        moving_avg = tf.train.ExponentialMovingAverage(0.9)
        eps = 1e-5
        if is_conv:
            mean, var = tf.nn.moments(input_tensor, [0, 1, 2])
        else:
            mean, var = tf.nn.moments(input_tensor, [0])
        update = moving_avg.apply([mean, var])
        self.tensor_updated.append(update)
        mean = tf.cond(self.is_train, lambda: mean, lambda: moving_avg.average(mean))
        var  = tf.cond(self.is_train, lambda: var, lambda: moving_avg.average(var))
        ret  = tf.nn.batch_normalization(input_tensor, mean, var, bias, scale, eps)
        return ret
        

    def bnn_conv2d_bn(self, name, input_tensor, kernel, stride, init_func = msra_init, bn_input = True, enable_scale = True, enable_bias = True, padding = "SAME"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.bnn_conv2d(input_tensor, kernel, stride, init_func, bn_input, padding)
            scale = None
            bias  = None
            if enable_scale:
                scale = ones_init([kernel[3]], "bn_scale")
            if enable_bias:
                bias  = zeros_init([kernel[3]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias) 
            return current_tensor


    def bnn_conv2d_bn_relu(self, name, input_tensor, kernel, stride, init_func = msra_init, bn_input = True, enable_scale = True, enable_bias = True, padding = "SAME"):
        current_tensor = self.bnn_conv2d_bn(name, input_tensor, kernel, stride, init_func, bn_input, enable_scale, enable_bias, padding)
        current_tensor = tf.nn.relu(current_tensor)
        return current_tensor


    def bnn_fc(self, input_tensor, shape, init_func = msra_init, enable_bn = False, enable_bias = True):
        W = init_func(shape, "fc_weights")
        W = binary(W)
        input_tensor = binary(input_tensor)
        current_tensor = tf.matmul(input_tensor, W)
        if enable_bias:
            bias = zeros_init([shape[1]], "fc_bias")
            current_tensor = tf.add(current_tensor, bias)
        if enable_bn:
            scale = ones_init([shape[1]], "bn_scale")
            bias  = zeros_init([shape[1]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias, is_conv = False) 
        return current_tensor

    
    def bnn_fc_relu(self, name, input_tensor, shape, init_func = msra_init, enable_bias = True, enable_bn = False, enable_dropout = False, rate = 0.2):

        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.bnn_fc(input_tensor, shape, init_func, enable_bn, enable_bias)
            current_tensor = tf.nn.relu(current_tensor)
            if enable_dropout:
                current_tensor = tf.nn.dropout(current_tensor, rate = rate)
            return current_tensor

