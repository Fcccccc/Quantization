import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.framework import ops

def msra_init(shape, name = "msra_weights"):
    gen = tf.truncated_normal_initializer(mean = 0.0, stddev = np.sqrt(2.0 / np.product(shape)))
    return tf.get_variable(name = name, shape = shape, initializer = gen)  

def zeros_init(shape, name = "zero_weights"):
    return tf.get_variable(name = name, shape = shape, initializer = tf.zeros_initializer())

def ones_init(shape, name = "one_weights"):
    return tf.get_variable(name = name, shape = shape, initializer = tf.ones_initializer())

def binary(input_tensor):
    g = tf.get_default_graph()
    with ops.name_scope("binary"):
        with g.gradient_override_map({"Sign":"Identity"}):
            x = tf.clip_by_value(input_tensor, -1, 1)
            return tf.sign(x)



def train(model, batch_size, data_handle, model_name = "default", weight_decay = 0.0005):
    X = model.X
    Y = model.Y
    result = model.result
    train  = model.Utils.is_train
    update = model.Utils.tensor_updated
    learning_rate = tf.placeholder(tf.float32)
    if not os.path.exists("log"):
        os.mkdir("log")
    fd = open("log/" + model_name, "a")
    print(time.asctime(time.localtime(time.time())) + "   train started", file = fd)

    cross_entropy     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = result))
    l2_loss           = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss              = l2_loss * weight_decay + cross_entropy
    # train_step        = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
    train_step        = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov = True).minimize(loss)
    top1              = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
    top1_acc          = tf.reduce_mean(tf.cast(top1, "float"))
    top5              = tf.nn.in_top_k(predictions = result, targets = tf.argmax(Y, 1), k = 5) 
    top5_acc          = tf.reduce_mean(tf.cast(top5, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        index = 0
        lr = 0.01
        for epoch in range(1, 125 + 1):
            gen = data_handle.epoch_data_train(batch_size)
            cnt = 0
            if epoch == 50: lr /= 10
            if epoch == 75: lr /= 10
            if epoch == 100: lr /= 10
            for x, y in gen():
                cnt += 1
                index += 1
                _, train_loss, train_acc, train_top5_acc, *skip = sess.run([train_step, loss, top1_acc, top5_acc] + update, feed_dict = {X: x,Y: y,train: True, learning_rate: lr})
                if cnt % 10 == 0:
                    print("train epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, train_loss, train_acc, train_top5_acc))
            if epoch % 5 == 0:
                x, y = data_handle.data_test()
                test_loss, test_acc, test_top5_acc = sess.run([loss, top1_acc, top5_acc], feed_dict = {X: x, Y: y, train: False})
                print("test epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, test_loss, test_acc, test_top5_acc))
                if epoch == 125:
                    print("loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(test_loss, test_acc, test_top5_acc), file = fd)
                    print(time.asctime(time.localtime(time.time())) + "   train finished",file =fd)
    fd.close()

