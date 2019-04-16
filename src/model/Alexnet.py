import sys
import os
sys.path.append(os.path.abspath(".."))
from Utils import cnnUtils
import tensorflow as tf
import numpy as np

class Alexnet:
    def __init__(self, input_shape, output_shape):
        self.X = tf.placeholder(dtype = tf.float32, shape = input_shape)
        self.Y = tf.placeholder(dtype = tf.float32, shape = output_shape)
        self.result   = None
        self.Utils = cnnUtils.nnUtils()

    def build(self):
        
        with tf.variable_scope("conv1", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(self.X, [3, 3, 3, 96], [2, 2])
            current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
            current_tensor = tf.nn.lrn(current_tensor)

        with tf.variable_scope("conv2", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 96, 256], [1, 1])
            current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
            current_tensor = tf.nn.lrn(current_tensor)

        with tf.variable_scope("conv3", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 256, 384], [1, 1])

        with tf.variable_scope("conv4", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 384, 384], [1, 1])

        with tf.variable_scope("conv5", reuse = tf.AUTO_REUSE):
            current_tensor = self.Utils.conv2d(current_tensor, [3, 3, 384, 256], [1, 1])

        current_tensor = tf.nn.max_pool(current_tensor, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID")
        current_tensor = tf.reshape(current_tensor, [-1, 1024])

        current_tensor = self.Utils.fc_relu("fc1", current_tensor, [1024, 4096])
        current_tensor = self.Utils.fc_relu("fc2", current_tensor, [4096, 4096])
        self.result    = self.Utils.fc(current_tensor, [4096, 100])

if __name__ == "__main__":
    Alexnet = Alexnet([None, 32, 32, 3], [None, 100])
    Alexnet.build()






