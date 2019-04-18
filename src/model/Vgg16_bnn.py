import sys
import os
from src.Utils import BNNUtils
import tensorflow as tf
import numpy as np

class Vgg16_bnn:
    def __init__(self, input_shape, output_shape):
        self.X = tf.placeholder(dtype = tf.float32, shape = input_shape)
        self.Y = tf.placeholder(dtype = tf.float32, shape = output_shape)
        self.result   = None
        self.Utils = BNNUtils.nnUtils()

    def build(self):
        current_tensor = self.Utils.bnn_conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 128], [1, 1], bn_input = False)
        current_tensor = self.Utils.bnn_conv2d_bn_relu("conv1_2", current_tensor, [3, 3, 128, 128], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.bnn_conv2d_bn_relu("conv2_2", current_tensor, [3, 3, 128, 256], [1, 1])
        current_tensor = self.Utils.bnn_conv2d_bn_relu("conv2_3", current_tensor, [3, 3, 256, 256], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.bnn_conv2d_bn_relu("conv3_1", current_tensor, [3, 3, 256, 512], [1, 1])
        current_tensor = self.Utils.bnn_conv2d_bn_relu("conv3_2", current_tensor, [3, 3, 512, 512], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        urrent_tensor = tf.reduce_mean(current_tensor, axis = [1, 2])

        current_tensor = tf.reshape(current_tensor, [-1, 16 * 512])

        current_tensor = self.Utils.bnn_fc_relu("fc1", current_tensor, [16 * 512, 1024], enable_bn = True)
        current_tensor = self.Utils.bnn_fc_relu("fc2", current_tensor, [1024, 1024], enable_bn = True)
        self.result    = self.Utils.bnn_fc(current_tensor, [1024, 100], enable_bn = True)

if __name__ == "__main__":
    vgg16_bnn = Vgg16_bnn([None, 32, 32, 3], [None, 100])
    vgg16_bnn.build()






