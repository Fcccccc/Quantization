

from src.Utils import cnnUtils
import tensorflow as tf

class normal_base:
    def __init__(self, hyperparams):
        self.X = tf.placeholder(dtype = tf.float32, shape = hyperparams['input_shape'])
        self.Y = tf.placeholder(dtype = tf.float32, shape = hyperparams['output_shape'])
        self.result   = None
        self.Utils = cnnUtils.cnnUtils()


