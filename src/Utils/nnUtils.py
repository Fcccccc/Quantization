import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.python.framework import ops
from tensorflow.contrib.model_pruning.python import pruning

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

def fake_quant(input_tensor, min, max, num_bits = 8):
    g = tf.get_default_graph()
    with ops.name_scope("fake_quant"):
        scale = (max - min) / (2 ** num_bits - 1)
        with g.gradient_override_map({"Round":"Identity"}):
            return tf.round(input_tensor / scale) * scale


def match(name, pattern):
    if pattern == None:
        return True
    pattern_len = len(pattern)
    for i in range(len(name) - pattern_len + 1):
        if name[i: i + pattern_len] == pattern:
            return True
    return False


def model_size(pattern = None):
    tot = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        if match(var.name, pattern):
            tmp = 1
            for dim in shape:
                tmp *= dim
            tot += tmp
    return tot


class Trainer:
    def __init__(self, model, data_handle, hyperparams):
        self.model = model
        self.data_handle = data_handle
        self.hyperparams = hyperparams

        # get defined tensor
        self.X = self.model.X
        self.Y = self.model.Y
        self.result = self.model.result
        self.train  = self.model.Utils.is_train
        self.update = self.model.Utils.tensor_updated
        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.train.get_or_create_global_step()
        self.weights_decay = self.hyperparams['weights_decay']

        # optimizer
        self.cross_entropy     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.result))
        self.l2_loss           = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss              = self.l2_loss * self.weights_decay + self.cross_entropy
        # train_step        = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
        self.train_step        = tf.train.MomentumOptimizer(self.learning_rate, 0.9, use_nesterov = True).minimize(self.loss)
        self.top1              = tf.equal(tf.argmax(self.result, 1), tf.argmax(self.Y, 1))
        self.top1_acc          = tf.reduce_mean(tf.cast(self.top1, "float"))
        self.top5              = tf.nn.in_top_k(predictions = self.result, targets = tf.argmax(self.Y, 1), k = 5) 
        self.top5_acc          = tf.reduce_mean(tf.cast(self.top5, "float"))


        # prune
        if self.hyperparams['enable_prune']:
            pruning_hparams = pruning.get_pruning_hparams()
            pruning_hparams.begin_pruning_step = self.hyperparams['begin_pruning_step']
            pruning_hparams.end_pruning_step   = self.hyperparams['end_pruning_step']
            pruning_hparams.pruning_frequency  = self.hyperparams['pruning_frequency']
            pruning_hparams.target_sparsity    = self.hyperparams['target_sparsity']
            p = pruning.Pruning(pruning_hparams, global_step = self.global_step, sparsity = pruning_hparams.target_sparsity)
            self.prune_op = p.conditional_mask_update_op()

        # log
        if not os.path.exists("log"):
            os.mkdir("log")
        self.fd = open("log/" + self.hyperparams['model_name'], "a")
        print("model_name = {}, quant_bits = {}, enable_prune = {}".format(self.hyperparams['model_name'], self.hyperparams['quant_bits'], self.hyperparams['enable_prune']), file = self.fd)
        print(time.asctime(time.localtime(time.time())) + "   train started", file = self.fd)


        # init_variable
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __del__(self):
        print("train successfully", file = self.fd)
        self.fd.close()
        self.sess.close()

        
    def train_all_epoch(self, batch_size, epoch_max, init_lr, lr_reduce, other_update = None):
        lr = init_lr
        for epoch in range(1, epoch_max + 1):
            gen = self.data_handle.epoch_data_train(batch_size)
            cnt = 0
            if epoch in lr_reduce:
                lr /= 10
            for x, y in gen():
                cnt += 1
                if other_update is not None:
                    self.sess.run(other_update)
                _, train_loss, train_acc, train_top5_acc, *skip = self.sess.run([self.train_step, self.loss, self.top1_acc, self.top5_acc] + self.update, feed_dict = {self.X: x,self.Y: y,self.train: True, self.learning_rate: lr})
                if cnt % 10 == 0:
                    print("train epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, train_loss, train_acc, train_top5_acc))
            if epoch % 5 == 0:
                x, y = self.data_handle.data_test()
                test_loss, test_acc, test_top5_acc = self.sess.run([self.loss, self.top1_acc, self.top5_acc], feed_dict = {self.X: x, self.Y: y, self.train: False})
                print("test epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, test_loss, test_acc, test_top5_acc))
                if epoch == epoch_max:
                    print("loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(test_loss, test_acc, test_top5_acc), file = self.fd)
                    print(time.asctime(time.localtime(time.time())) + "   train finished",file = self.fd)

    def dump_weights(self):
        train_var_name = tf.trainable_variables()
        total_var      = []
        prefix = self.hyperparams['model_name'] + "_" + str(self.hyperparams['quant_bits']) + "_" + str(self.hyperparams['target_sparsity']) + "/"
        for var_name in train_var_name:
            total_var.append(tf.get_default_graph().get_tensor_by_name(var_name.name))
        for tensor in total_var:
            file_name = prefix + tensor.name + "_float32"
            data = self.sess.run(tensor)
            data = np.reshape(data, (-1))
            if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))
            with open(file_name, "w") as fd:
                for elem in data:
                    print(elem, file = fd)
            if self.hyperparams['enable_quant']:
                tmp_data = fake_quant(tensor, min = tf.reduce_min(tensor), max = tf.reduce_max(tensor), num_bits = self.hyperparams['quant_bits'])
                quant_data = self.sess.run(tmp_data)
                quant_data = np.reshape(quant_data, -1)
                quant_file_name = prefix + tensor.name + "_quant"
                with open(quant_file_name, "w") as fd:
                    for elem in quant_data:
                        print(elem, file = fd)


    def do_train(self):
        self.train_all_epoch(self.hyperparams['train_batchsize'], self.hyperparams['train_epoch'], self.hyperparams['train_lr'], self.hyperparams['train_lr_reduce'])
        if self.hyperparams['enable_prune']:
            self.train_all_epoch(self.hyperparams['prune_batchsize'], self.hyperparams['prune_epoch'], self.hyperparams['prune_lr'], self.hyperparams['prune_lr_reduce'], other_update = self.prune_op)
        if self.hyperparams['enable_dump_weights']:
            self.dump_weights()
        
        


# def train(model, batch_size, data_handle, model_name = "default", enable_prune = False):
    # hypeparams = {
            # "weights_decay":0.0005,
            # "lr":0.01,
            # "enable_prune":False
                    # }

    # X = model.X
    # Y = model.Y
    # result = model.result
    # train  = model.Utils.is_train
    # update = model.Utils.tensor_updated
    # learning_rate = tf.placeholder(tf.float32)
    # global_step = tf.train.get_or_create_global_step()
    # weight_decay = 0.0005

    # if not os.path.exists("log"):
        # os.mkdir("log")
    # fd = open("log/" + model_name, "a")
    # print(time.asctime(time.localtime(time.time())) + "   train started", file = fd)

    # dump weights by tensor name
    # train_var_name = tf.trainable_variables()
    # total_var      = []
    # for var_name in train_var_name:
        # total_var.append(tf.get_default_graph().get_tensor_by_name(var_name.name))



    # cross_entropy     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = result))
    # l2_loss           = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    # loss              = l2_loss * weight_decay + cross_entropy
    # train_step        = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
    # train_step        = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov = True).minimize(loss)
    # top1              = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
    # top1_acc          = tf.reduce_mean(tf.cast(top1, "float"))
    # top5              = tf.nn.in_top_k(predictions = result, targets = tf.argmax(Y, 1), k = 5) 
    # top5_acc          = tf.reduce_mean(tf.cast(top5, "float"))


    # if enable_prune:
        # pruning_hparams = pruning.get_pruning_hparams()
        # pruning_hparams.begin_pruning_step = 0
        # pruning_hparams.end_pruning_step   = -1
        # pruning_hparams.pruning_frequency  = 5
        # pruning_hparams.target_sparsity    = 0.6
        # p = pruning.Pruning(pruning_hparams, global_step = global_step, sparsity = 0.6)
        # prune_op = p.conditional_mask_update_op()
    # with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # index = 0
        # lr = 0.01
        # for epoch in range(1, 125 + 1):
            # gen = data_handle.epoch_data_train(batch_size)
            # cnt = 0
            # if epoch == 50: lr /= 10
            # if epoch == 75: lr /= 10
            # if epoch == 100: lr /= 10
            # for x, y in gen():
                # cnt += 1
                # index += 1
                # _, train_loss, train_acc, train_top5_acc, *skip = sess.run([train_step, loss, top1_acc, top5_acc] + update, feed_dict = {X: x,Y: y,train: True, learning_rate: lr})
                # if cnt % 10 == 0:
                    # print("train epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, train_loss, train_acc, train_top5_acc))
            # if epoch % 5 == 0:
                # x, y = data_handle.data_test()
                # test_loss, test_acc, test_top5_acc = sess.run([loss, top1_acc, top5_acc], feed_dict = {X: x, Y: y, train: False})
                # print("test epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, test_loss, test_acc, test_top5_acc))
                # if epoch == 125:
                    # print("loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(test_loss, test_acc, test_top5_acc), file = fd)
                    # print(time.asctime(time.localtime(time.time())) + "   train finished",file =fd)

        # if enable_prune:
            # print(time.asctime(time.localtime(time.time())) + "   prune started",file =fd)
            # lr = 0.0001
            # for epoch in range(1, 50 + 1):
                # gen = data_handle.epoch_data_train(batch_size)
                # cnt = 0
                # if epoch == 25: lr /= 10
                # for x, y in gen():
                    # cnt += 1
                    # index += 1
                    # sess.run(prune_op)
                    # _, train_loss, train_acc, train_top5_acc, *skip = sess.run([train_step, loss, top1_acc, top5_acc] + update, feed_dict = {X: x,Y: y,train: True, learning_rate: lr})
                    # if cnt % 10 == 0:
                        # print("train epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, train_loss, train_acc, train_top5_acc))
                # if epoch % 5 == 0:
                    # x, y = data_handle.data_test()
                    # test_loss, test_acc, test_top5_acc = sess.run([loss, top1_acc, top5_acc], feed_dict = {X: x, Y: y, train: False})
                    # print("test epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, test_loss, test_acc, test_top5_acc))
                    # if epoch == 50:
                        # print("loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(test_loss, test_acc, test_top5_acc), file = fd)
                        # print(time.asctime(time.localtime(time.time())) + "   prune finished",file =fd)

        # if enable_dump_weights:
            # for tensor in total_var:
                # name = "data/" + tensor.name
                # data = sess.run(tensor)
                # data = np.reshape(data, (-1))
                # if not os.path.exists(os.path.dirname(name)):
                    # os.makedirs(os.path.dirname(name))
                # with open(name, "w") as fd:
                    # for elem in data:
                        # print(elem, file = fd)


    # fd.close()

