import sys
import os
from src.quant_model.base import quant_base
import tensorflow as tf
import numpy as np

class Vgg16(quant_base):
    def __init__(self, hyperparams):
        quant_base.__init__(self, hyperparams)

    def build(self, quant_bits = 8):
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 64], [1, 1], quant_bits = 8)
        current_tensor = self.Utils.quant_conv2d_bn_relu("conv1_2", current_tensor, [3, 3, 64, 64], [1, 1], quant_bits = 8)
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
        self.result    = self.Utils.quant_fc(current_tensor, [4096, 100], quant_bits = self.quant_bits)

if __name__ == "__main__":
    vgg16 = Vgg16({
        "input_shape":[None, 32, 32, 3],
        "output_shape":[None, 100],
        "quant_bits":4})
    vgg16.build()






