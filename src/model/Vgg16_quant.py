import sys
import os
sys.path.append(os.path.abspath(".."))
from Utils import cnnUtils
import tensorflow as tf
import numpy as np

class Vgg16_quant:
    def __init__(self, input_shape, output_shape, quant_bits = 8):
        self.X = tf.placeholder(dtype = tf.float32, shape = input_shape)
        self.Y = tf.placeholder(dtype = tf.float32, shape = output_shape)
        self.result   = None
        self.Utils = cnnUtils.nnUtils()
        self.quant_bits = quant_bits

    def build(self, quant_bits = 8):
        current_tensor = self.Utils.conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 64], [1, 1])
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv1_2", current_tensor, [3, 3, 64, 64], [1, 1], quant_bits = self.quant_bits)
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.quant_conv2d_bn_relu("conv2_1", current_tensor, [3, 3, 64, 128], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv2_2", current_tensor, [3, 3, 128, 128], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv2_3", current_tensor, [1, 1, 128, 128], [1, 1], quant_bits = self.quant_bits)
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.quant_conv2d_bn_relu("conv3_1", current_tensor, [3, 3, 128, 256], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv3_2", current_tensor, [3, 3, 256, 256], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv3_3", current_tensor, [1, 1, 256, 256], [1, 1], quant_bits = self.quant_bits)
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.quant_conv2d_bn_relu("conv4_1", current_tensor, [3, 3, 256, 512], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv4_2", current_tensor, [3, 3, 512, 512], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv4_3", current_tensor, [1, 1, 512, 512], [1, 1], quant_bits = self.quant_bits)
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.quant_conv2d_bn_relu("conv5_1", current_tensor, [3, 3, 512, 512], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv5_2", current_tensor, [3, 3, 512, 512], [1, 1], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv5_3", current_tensor, [1, 1, 512, 512], [1, 1], quant_bits = self.quant_bits)
        current_tensor = tf.reduce_mean(current_tensor, axis = [1, 2])

        current_tensor = tf.reshape(current_tensor, [-1, 512])

        current_tensor = self.Utils.quant_fc_relu("fc1", current_tensor, [512, 4096], quant_bits = self.quant_bits)
        current_tensor = self.Utils.quant_fc_relu("fc2", current_tensor, [4096, 4096], quant_bits = self.quant_bits)
        self.result    = self.Utils.fc(current_tensor, [4096, 100])

if __name__ == "__main__":
    vgg16_quant = Vgg16_quant([None, 32, 32, 3], [None, 100], quant_bits = 4)
    vgg16_quant.build()






