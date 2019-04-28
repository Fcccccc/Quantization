import sys
import os
from src.Utils import cnnUtils
from src.normal_model.base import normal_base
import tensorflow as tf
import numpy as np

class Resnet18(normal_base):
    def __init__(self, hyperparams):
        normal_base.__init__(self, hyperparams)

    def build(self):
        current_tensor = self.Utils.conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 64], [1, 1])

        current_tensor = self._residual_block(current_tensor, "res_block1", [3, 3, 64, 64])
        current_tensor = self._residual_block(current_tensor, "res_block2", [3, 3, 64, 64])
        current_tensor = self._residual_block(current_tensor, "res_block3", [3, 3, 64, 128], div = True)
        current_tensor = self._residual_block(current_tensor, "res_block4", [3, 3, 128, 128])
        current_tensor = self._residual_block(current_tensor, "res_block5", [3, 3, 128, 256], div = True)
        current_tensor = self._residual_block(current_tensor, "res_block6", [3, 3, 256, 256])
        current_tensor = self._residual_block(current_tensor, "res_block7", [3, 3, 256, 512], div = True)
        current_tensor = self._residual_block(current_tensor, "res_block8", [3, 3, 512, 512])

        current_tensor = tf.reduce_mean(current_tensor, axis = [1, 2])

        current_tensor = tf.reshape(current_tensor, [-1, 512])

        self.result    = self.Utils.fc(current_tensor, [512, 100])

    def _residual_block(self, input_tensor, name, kernel, div = False):
        if div:
            current_tensor = self.Utils.conv2d_bn_relu(name + "_conv1", input_tensor, kernel, [2, 2])
            shortcut = self.Utils.conv2d_bn(name + "_shortcut", input_tensor, kernel, [2, 2])
        else:
            current_tensor = self.Utils.conv2d_bn_relu(name + "_conv1", input_tensor, kernel, [1, 1])
            shortcut = input_tensor
        kernel[2] = kernel[3]
        current_tensor = self.Utils.conv2d_bn(name + "_conv2", current_tensor, kernel, [1, 1])
        current_tensor = tf.add(current_tensor, shortcut)
        return tf.nn.relu(current_tensor)

   
if __name__ == "__main__":
    resnet18 = Resnet18({
        "input_shape":[None, 32, 32, 3],
        "output_shape":[None, 100]})
 
    resnet18.build()






