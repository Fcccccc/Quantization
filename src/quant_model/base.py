import tensorflow as tf
from src.Utils import cnnUtils
class quant_base:
    def __init__(self, hyperparams):
        self.X = tf.placeholder(dtype = tf.float32, shape = hyperparams['input_shape'])
        self.Y = tf.placeholder(dtype = tf.float32, shape = hyperparams['output_shape'])
        self.result   = None
        self.Utils = cnnUtils.cnnUtils()
        self.quant_bits = hyperparams['quant_bits']


