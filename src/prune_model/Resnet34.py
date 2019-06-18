import sys
import os
from src.Utils import cnnUtils
import tensorflow as tf
from src.normal_model.base import normal_base
import numpy as np

class Resnet34(normal_base):

    def __init__(self, hyperparams):
        normal_base.__init__(self, hyperparams)

    def build(self):
        current_tensor = self.Utils.conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 64], [1, 1], enable_prune = True)

        current_tensor = self._residual_group(current_tensor, "res_group0", 64, 64, cnt = 3, enable_prune = True)
        current_tensor = self._residual_group(current_tensor, "res_group1", 64, 128, div = True, cnt = 4, enable_prune = True)
        current_tensor = self._residual_group(current_tensor, "res_group2", 128, 256, div = True, cnt = 6, enable_prune = True)
        current_tensor = self._residual_group(current_tensor, "res_group3", 256, 512, div = True, cnt = 3, enable_prune = True)

        current_tensor = tf.reduce_mean(current_tensor, axis = [1, 2])

        current_tensor = tf.reshape(current_tensor, [-1, 512])

        self.result    = self.Utils.fc(current_tensor, [512, 100], enable_prune = True)



    def _residual_group(self, input_tensor, name, input_channel, output_channel, cnt, div = False, enable_prune = False):
        current_tensor = self._residual_block(input_tensor, name + "_block0", [input_channel, output_channel], div = div)
        for i in range(1, cnt):
            current_tensor = self._residual_block(current_tensor, name + "_block" + str(i), [input_channel, output_channel], enable_prune = enable_prune)
        return current_tensor

    def _residual_block(self, input_tensor, name, kernel, div = False, enable_prune = False):
        input_channel = kernel[0]
        output_channel = kernel[1]
        if div:
            current_tensor = self.Utils.conv2d_bn_relu(name + "_conv0", input_tensor, [3, 3, input_channel, output_channel], [2, 2], enable_prune = True)
            shortcut = self.Utils.conv2d_bn(name + "_shortcut", input_tensor, [3, 3, input_channel, output_channel], [2, 2])
        else:
            current_tensor = self.Utils.conv2d_bn_relu(name + "_conv0", input_tensor, [3, 3, output_channel, output_channel], [1, 1], enable_prune = True)
            shortcut = input_tensor
        current_tensor = self.Utils.conv2d_bn(name + "_conv1", current_tensor, [3, 3, output_channel, output_channel], [1, 1], enable_prune = enable_prune)
        current_tensor = tf.add(current_tensor, shortcut)
        return tf.nn.relu(current_tensor)

if __name__ == "__main__":
    resnet34 = Resnet34({
        "input_shape":[None, 32, 32, 3],
        "output_shape":[None, 100]})
    resnet34.build()






