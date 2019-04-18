import tensorflow as tf
import numpy as np
import time
import os
import sys
from src.Utils.nnUtils import msra_init, ones_init, zeros_init


class cnnUtils:

    def __init__(self):
        self.is_train = tf.placeholder(dtype = tf.bool)
        self.tensor_updated = []

    def conv2d(self, input_tensor, kernel, stride, init_func = msra_init, padding = "SAME"):
        assert(len(kernel) == 4)
        assert(len(stride) == 2)
        W = init_func(kernel, "conv_weights")
        conv_res = tf.nn.conv2d(input_tensor, W, [1, stride[0], stride[1], 1], padding = padding)
        return conv_res
        
    def quant_conv2d(self, input_tensor, kernel, stride, init_func = msra_init, quant_bits = 8, padding = "SAME", quant_input = True):
        assert(len(kernel) == 4)
        assert(len(stride) == 2)
        W = init_func(kernel, "conv_weights")
        W = tf.fake_quant_with_min_max_vars(W, min = tf.reduce_min(W), max = tf.reduce_max(W), num_bits = quant_bits)
        if quant_input == True:
            input_tensor = tf.fake_quant_with_min_max_vars(input_tensor, min = tf.reduce_min(input_tensor), max = tf.reduce_max(input_tensor), num_bits = quant_bits)
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
        

    def conv2d_bn(self, name, input_tensor, kernel, stride, init_func = msra_init, enable_scale = True, enable_bias = True, padding = "SAME"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.conv2d(input_tensor, kernel, stride, init_func, padding)
            scale = None
            bias  = None
            if enable_scale:
                scale = ones_init([kernel[3]], "bn_scale")
            if enable_bias:
                bias  = zeros_init([kernel[3]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias) 
            return current_tensor


    def conv2d_bn_relu(self, name, input_tensor, kernel, stride, init_func = msra_init, enable_scale = True, enable_bias = True, padding = "SAME"):
        current_tensor = self.conv2d_bn(name, input_tensor, kernel, stride, init_func, enable_scale, enable_bias, padding)
        current_tensor = tf.nn.relu(current_tensor)
        return current_tensor

    def quant_conv2d_bn(self, name, input_tensor, kernel, stride, init_func = msra_init, quant_bits = 8, enable_scale = True, enable_bias = True):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.quant_conv2d(input_tensor, kernel, stride, init_func, quant_bits = quant_bits)
            scale = None
            bias  = None
            if enable_scale:
                scale = ones_init([kernel[3]], "bn_scale")
            if enable_bias:
                bias  = zeros_init([kernel[3]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias)
            return current_tensor

    def quant_conv2d_bn_relu(self, name, input_tensor, kernel, stride, init_func = msra_init, quant_bits = 8, enable_scale = True, enable_bias = True):
        current_tensor = self.quant_conv2d_bn(name, input_tensor, kernel, stride, init_func, quant_bits, enable_scale, enable_bias)
        current_tensor = tf.nn.relu(current_tensor)
        return current_tensor


    def fc(self, input_tensor, shape, init_func = msra_init, enable_bn = False, enable_bias = True):
        W = init_func(shape, "fc_weights")
        current_tensor = tf.matmul(input_tensor, W)
        if enable_bias:
            bias = zeros_init([shape[1]], "fc_bias")
            current_tensor = tf.add(current_tensor, bias)
        if enable_bn:
            scale = ones_init([shape[1]], "bn_scale")
            bias  = zeros_init([shape[1]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias, is_conv = False) 
        return current_tensor

    def quant_fc(self, input_tensor, shape, init_func = msra_init, enable_bias = True, quant_bits = 8, quant_input = True):
        W = init_func(shape, "fc_weights")
        W = tf.fake_quant_with_min_max_vars(W, min = tf.reduce_min(W), max = tf.reduce_max(W), num_bits = quant_bits)
        if quant_input == True:
            input_tensor = tf.fake_quant_with_min_max_vars(input_tensor, min = tf.reduce_min(input_tensor), max = tf.reduce_max(input_tensor), num_bits = quant_bits)
        current_tensor = tf.matmul(input_tensor, W)
        if enable_bias:
            bias = zeros_init([shape[1]], "fc_bias")
            current_tensor = tf.add(current_tensor, bias)
        return current_tensor

    def fc_relu(self, name, input_tensor, shape, init_func = msra_init, enable_bias = True, enable_bn = False, enable_dropout = False, rate = 0.2):

        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.fc(input_tensor, shape, init_func, enable_bn, enable_bias)
            current_tensor = tf.nn.relu(current_tensor)
            if enable_dropout:
                current_tensor = tf.nn.dropout(current_tensor, rate = rate)
            return current_tensor

    def quant_fc_relu(self, name, input_tensor, shape, init_func = msra_init, enable_bias = True, quant_bits = 8, quant_input = True, enable_dropout = False, rate = 0.2):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.quant_fc(input_tensor, shape, init_func, enable_bias, quant_bits, quant_input)
            current_tensor = tf.nn.relu(current_tensor)
            if enable_dropout:
                current_tensor = tf.nn.dropout(current_tensor, rate = rate)
            return current_tensor

