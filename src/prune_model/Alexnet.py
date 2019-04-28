import sys
import os
from src.Utils import cnnUtils
from src.normal_model.base import normal_base
import tensorflow as tf
import numpy as np

class Alexnet(normal_base):
    def __init__(self, hyperparams):
        normal_base.__init__(self, hyperparams)

    def build(self):
        
        with tf.variable_scope("conv1", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(self.X, [3, 3, 3, 64], [2, 2])
            current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
            current_tensor = tf.nn.lrn(current_tensor)

        with tf.variable_scope("conv2", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 64, 64], [1, 1])
            current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
            current_tensor = tf.nn.lrn(current_tensor)

        # current_tensor = self.Utils.conv2d_bn_relu("conv1", self.X, [3, 3, 3, 96], [2, 2])
        # current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
        # current_tensor = self.Utils.conv2d_bn_relu("conv2", current_tensor, [3, 3, 96, 256], [1, 1])
        # current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")

        
        current_tensor = self.Utils.conv2d_bn_relu("conv3", current_tensor, [3, 3, 64, 128], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv4", current_tensor, [3, 3, 128, 128], [1, 1])
        current_tensor = self.Utils.conv2d_bn_relu("conv5", current_tensor, [3, 3, 128, 128], [1, 1], enable_prune = True)


        # with tf.variable_scope("conv3", reuse = tf.AUTO_REUSE):
            # current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 64, 128], [1, 1])
            # current_tensor = tf.nn.relu(current_tensor)
            # self.debug.append(current_tensor)

        # with tf.variable_scope("conv4", reuse = tf.AUTO_REUSE):
            # current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 128, 128], [1, 1])
            # current_tensor = tf.nn.relu(current_tensor)
            # self.debug.append(current_tensor)

        # with tf.variable_scope("conv5", reuse = tf.AUTO_REUSE):
            # current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 128, 128], [1, 1])
            # current_tensor = tf.nn.relu(current_tensor)
            # self.debug.append(current_tensor)

        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
        current_tensor = tf.reshape(current_tensor, [-1, 512])

        current_tensor = self.Utils.fc_relu("fc1", current_tensor, [512, 1024], enable_prune = True)
        current_tensor = self.Utils.fc_relu("fc2", current_tensor, [1024, 1024], enable_prune = True)
        self.result    = self.Utils.fc(current_tensor, [1024, 100], enable_prune = True)

if __name__ == "__main__":

    Alexnet = Alexnet({
        "input_shape":[None, 32, 32, 3],
        "output_shape":[None, 100]})
    Alexnet.build()






