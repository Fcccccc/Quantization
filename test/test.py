import tensorflow as tf
import numpy as np



def quant(x):
    g = tf.get_default_graph()
    with g.
    return tf.multiply(x, 2.0)

def main():
    X = tf.placeholder(shape = [1], dtype = tf.float32)
    Y = tf.placeholder(shape = [1], dtype = tf.float32)
    w = tf.Variable([2], dtype = tf.float32)
    b = tf.Variable([3], dtype = tf.float32)
    quant_w = tf.identity(w)
    quant_w = quant(w)
    quant_b = tf.identity(b)
    quant_b = quant(b)
    

if __name__ == "__main__":
    main()
