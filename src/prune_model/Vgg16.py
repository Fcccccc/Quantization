import sys
import os
from src.Utils import cnnUtils
import tensorflow as tf
import numpy as np
from tensorflow.contrib.model_pruning.python import pruning
from src.normal_model.base import normal_base

class Vgg16(normal_base):
    def __init__(self, hyperparams):
        normal_base.__init__(self, hyperparams)

    def build(self):
        current_tensor = self.Utils.conv2d_bn_relu("conv1_1", self.X, [3, 3, 3, 64], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv1_2", current_tensor, [3, 3, 64, 64], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.conv2d_bn_relu("conv2_1", current_tensor, [3, 3, 64, 128], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv2_2", current_tensor, [3, 3, 128, 128], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv2_3", current_tensor, [1, 1, 128, 128], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.conv2d_bn_relu("conv3_1", current_tensor, [3, 3, 128, 256], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv3_2", current_tensor, [3, 3, 256, 256], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv3_3", current_tensor, [1, 1, 256, 256], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.conv2d_bn_relu("conv4_1", current_tensor, [3, 3, 256, 512], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv4_2", current_tensor, [3, 3, 512, 512], [1, 1], enable_prune = True)
        current_tensor = self.Utils.conv2d_bn_relu("conv4_3", current_tensor, [1, 1, 512, 512], [1, 1])
        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        current_tensor = self.Utils.conv2d_bn_relu("conv5_1", current_tensor, [3, 3, 512, 512], [1, 1], enable_prune = True)
        current_tensor = self.Utils.conv2d_bn_relu("conv5_2", current_tensor, [3, 3, 512, 512], [1, 1], enable_prune = True)
        current_tensor = self.Utils.conv2d_bn_relu("conv5_3", current_tensor, [1, 1, 512, 512], [1, 1], enable_prune = True)
        current_tensor = tf.reduce_mean(current_tensor, axis = [1, 2])

        current_tensor = tf.reshape(current_tensor, [-1, 512])

        current_tensor = self.Utils.fc_relu("fc1", current_tensor, [512, 4096], enable_bn = True, enable_prune = True)
        current_tensor = self.Utils.fc_relu("fc2", current_tensor, [4096, 4096], enable_bn = True, enable_prune = True)
        self.result    = self.Utils.fc(current_tensor, [4096, 100], enable_prune = True)

if __name__ == "__main__":
    vgg16 = Vgg16({
        "input_shape":[None, 32, 32, 3],
        "output_shape":[None, 100]})
    vgg16.build()






