import sys
import os
sys.path.append(os.path.abspath(".."))
from Utils import cnnUtils
import tensorflow as tf
import numpy as np

class Resnet18_quant:
    def __init__(self, input_shape, output_shape):
        self.X = tf.placeholder(dtype = tf.float32, shape = input_shape)
        self.Y = tf.placeholder(dtype = tf.float32, shape = output_shape)
        self.result   = None
        self.Utils = cnnUtils.nnUtils()

    def build(self, quant_bits = 8):
        current_tensor = self.Utils.conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 64], [1, 1])

        current_tensor = self._quant_residual_block(current_tensor, "res_block1", [3, 3, 64, 64], quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block2", [3, 3, 64, 64], quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block3", [3, 3, 64, 128], div = True, quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block4", [3, 3, 128, 128], quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block5", [3, 3, 128, 256], div = True, quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block6", [3, 3, 256, 256], quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block7", [3, 3, 256, 512], div = True, quant_bits = quant_bits)
        current_tensor = self._quant_residual_block(current_tensor, "res_block8", [3, 3, 512, 512], quant_bits = quant_bits)

        current_tensor = tf.reduce_mean(current_tensor, axis = [1, 2])

        current_tensor = tf.reshape(current_tensor, [-1, 512])

        self.result    = self.Utils.fc(current_tensor, [512, 100])

    def _quant_residual_block(self, input_tensor, name, kernel, div = False, quant_bits = 8):
        if div:
            current_tensor = self.Utils.quant_conv2d_bn_relu(name + "_conv1", input_tensor, kernel, [2, 2], quant_bits = quant_bits)
            shortcut = self.Utils.quant_conv2d_bn(name + "_shortcut", input_tensor, kernel, [2, 2], quant_bits = quant_bits)
        else:
            current_tensor = self.Utils.quant_conv2d_bn_relu(name + "_conv1", input_tensor, kernel, [1, 1], quant_bits = quant_bits)
            shortcut = input_tensor
        kernel[2] = kernel[3]
        current_tensor = self.Utils.quant_conv2d_bn(name + "_conv2", current_tensor, kernel, [1, 1], quant_bits = quant_bits)
        current_tensor = tf.add(current_tensor, shortcut)
        return tf.nn.relu(current_tensor)

   
if __name__ == "__main__":
    resnet18_quant = Resnet18_quant([None, 32, 32, 3], [None, 100])
    resnet18_quant.build(quant_bits = 4)





