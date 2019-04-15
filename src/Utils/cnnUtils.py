import tensorflow as tf
import numpy as np
import time
import os

def msra_init(shape, name = "msra_weights"):
    gen = tf.truncated_normal_initializer(mean = 0.0, stddev = np.sqrt(2.0 / np.product(shape)))
    return tf.get_variable(name = name, shape = shape, initializer = gen)  

def zeros_init(shape, name = "zero_weights"):
    return tf.get_variable(name = name, shape = shape, initializer = tf.zeros_initializer())

def ones_init(shape, name = "one_weights"):
    return tf.get_variable(name = name, shape = shape, initializer = tf.ones_initializer())




class nnUtils:

    def __init__(self):
        self.is_train = tf.placeholder(dtype = tf.bool)
        self.tensor_updated = []

    def conv2d(self, input_tensor, kernel, stride, init_func = msra_init, padding = "SAME"):
        assert(len(kernel) == 4)
        assert(len(stride) == 2)
        W = init_func(kernel, "conv_weights")
        conv_res = tf.nn.conv2d(input_tensor, W, [1, stride[0], stride[1], 1], padding = padding)
        return conv_res
        
    def quant_conv2d(self, input_tensor, kernel, stride, init_func = msra_init, quant_bits = 8, padding = "SAME", quant_input = True):
        assert(len(kernel) == 4)
        assert(len(stride) == 2)
        W = init_func(kernel, "conv_weights")
        W = tf.fake_quant_with_min_max_vars(W, min = tf.reduce_min(W), max = tf.reduce_max(W), num_bits = quant_bits)
        if quant_input == True:
            input_tensor = tf.fake_quant_with_min_max_vars(input_tensor, min = tf.reduce_min(input_tensor), max = tf.reduce_max(input_tensor), num_bits = quant_bits)
        conv_res = tf.nn.conv2d(input_tensor, W, [1, stride[0], stride[1], 1], padding = padding)
        return conv_res

    def bn(self, input_tensor, scale, bias, is_conv = True):
        moving_avg = tf.train.ExponentialMovingAverage(0.9)
        eps = 1e-5
        if is_conv:
            mean, var = tf.nn.moments(input_tensor, [0, 1, 2])
        else:
            mean, var = tf.nn.moments(input_tensor, [0])
        update = moving_avg.apply([mean, var])
        self.tensor_updated.append(update)
        mean = tf.cond(self.is_train, lambda: mean, lambda: moving_avg.average(mean))
        var  = tf.cond(self.is_train, lambda: var, lambda: moving_avg.average(var))
        ret  = tf.nn.batch_normalization(input_tensor, mean, var, bias, scale, eps)
        return ret
        

    def conv2d_bn(self, name, input_tensor, kernel, stride, init_func = msra_init, enable_scale = True, enable_bias = True, padding = "SAME"):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.conv2d(input_tensor, kernel, stride, init_func, padding)
            scale = None
            bias  = None
            if enable_scale:
                scale = ones_init([kernel[3]], "bn_scale")
            if enable_bias:
                bias  = zeros_init([kernel[3]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias) 
            return current_tensor


    def conv2d_bn_relu(self, name, input_tensor, kernel, stride, init_func = msra_init, enable_scale = True, enable_bias = True, padding = "SAME"):
        current_tensor = self.conv2d_bn(name, input_tensor, kernel, stride, init_func, enable_scale, enable_bias, padding)
        current_tensor = tf.nn.relu(current_tensor)
        return current_tensor

    def quant_conv2d_bn_relu(self, name, input_tensor, kernel, stride, init_func = msra_init, quant_bits = 8, enable_scale = True, enable_bias = True):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.quant_conv2d(input_tensor, kernel, stride, init_func, quant_bits = quant_bits)
            scale = None
            bias  = None
            if enable_scale:
                scale = ones_init([kernel[3]], "bn_scale")
            if enable_bias:
                bias  = zeros_init([kernel[3]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias) 
            return current_tensor


    def fc(self, input_tensor, shape, init_func = msra_init, enable_bn = False, enable_bias = True):
        W = init_func(shape, "fc_weights")
        current_tensor = tf.matmul(input_tensor, W)
        if enable_bias:
            bias = zeros_init([shape[1]], "fc_bias")
            current_tensor = tf.add(current_tensor, bias)
        if enable_bn:
            scale = ones_init([shape[1]], "bn_scale")
            bias  = zeros_init([shape[1]], "bn_bias")
            current_tensor = self.bn(current_tensor, scale, bias, is_conv = False) 
        return current_tensor

    def quant_fc(self, input_tensor, shape, init_func = msra_init, enable_bias = True, quant_bits = 8, quant_input = True):
        W = init_func(shape, "fc_weights")
        W = tf.fake_quant_with_min_max_vars(W, min = tf.reduce_min(W), max = tf.reduce_max(W), num_bits = quant_bits)
        if quant_input == True:
            input_tensor = tf.fake_quant_with_min_max_vars(input_tensor, min = tf.reduce_min(input_tensor), max = tf.reduce_max(input_tensor), num_bits = quant_bits)
        current_tensor = tf.matmul(input_tensor, W)
        if enable_bias:
            bias = zeros_init([shape[1]], "fc_bias")
            current_tensor = tf.add(current_tensor, bias)
        return current_tensor

    def fc_relu(self, name, input_tensor, shape, init_func = msra_init, enable_bias = True, enable_bn = False):

        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.fc(input_tensor, shape, init_func, enable_bn, enable_bias)
            return tf.nn.relu(current_tensor)

    def quant_fc_relu(self, name, input_tensor, shape, init_func = msra_init, enable_bias = True, quant_bits = 8, quant_input = True):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            current_tensor = self.quant_fc(input_tensor, shape, init_func, enable_bias, quant_bits, quant_input)
            return tf.nn.relu(current_tensor)


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



if __name__ == "__main__":
    smsra = nnUtitls().msra_init([3, 3, 3])
    zero = nnUtitls().zeros_init([3, 3, 3])
    one  = nnUtitls().ones_init([3, 3, 3])
    
