#1 Q: if is_training = Ture and test model, will tf.contrib.batch_norm do bp algo?
#2 Q: updates_collection?
#3 Q: how to write code about BN?
#4 Q: differents initializer and how to write the customal initializer?
#5 Q: Optimizer and how to write customal Optimizer
#6 Q: L1, L2 regulaiztion


## should remember ...

# tf.nn.opr  the basical API
# tf.variable_scope   get_variable()  reuse = tf.AUTO_REUSE through function
# tf.trainable_variables 
# tf.graph()  interesting. it can define more than one computating Graph separating the opr and computing resource. May be it's useful for parallism. tf.Session() also can choose the graph to compute, but it have to pass the handle of Graph.
# CUDA VISIABLE DEVICE change the default GPU


# tf.train.AdamOptimizer top1_acc 68.15 top5_acc 88.87  1 x 1 conv not used.. :q
# tf.train.MomentumOptimizer top1_acc 71.13 top5_acc 91.27
# self.bn. Momentumoptimizer top1_acc 66 top5_acc 88  diffirent bn result in different accurancy
# different optimizer make sense?

# net structure, mostly conv...  
# three fc is better?




import tensorflow as tf
import numpy as np
import keras
import data_manager



class Vgg:
    
    
    def __init__(self):
        self.loss        = 0
        self.train_step  = 0
        self.top1_acc    = 0
        self.top5_acc    = 0
        self.X           = 0
        self.Y           = 0
        self.lr          = 0

    ## 
        self.weight_decay = 1e-4
        self.batch_size   = 128
        self.is_training  = tf.placeholder(dtype = "bool")
        self.bn_group     = []

    def bn(self, input_tensor, scale, bias, conv = True):
        moving_avg = tf.train.ExponentialMovingAverage(0.9)
        eps = 1e-5
        if conv:
            mean, var = tf.nn.moments(input_tensor, [0, 1, 2])
        else:
            mean, var = tf.nn.moments(input_tensor, [0])
        update = moving_avg.apply([mean, var])
        self.bn_group.append(update)
        mean = tf.cond(self.is_training, lambda: mean, lambda: moving_avg.average(mean))
        var  = tf.cond(self.is_training, lambda: var, lambda: moving_avg.average(var))
        ret  = tf.nn.batch_normalization(input_tensor, mean, var, bias, scale, eps)
        return ret
    def weights_msra(self, shape, name = "weights_msra"):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.variance_scaling_initializer())

    def weights_xavier(self, shape, name = "weights_xavier"):
        return tf.get_variable(name = name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

    def bias(self, shape, name = "bias"):
        initial = tf.constant(0.0,shape = shape)
        return tf.get_variable(name = name, initializer = initial)

    def conv2d(self, input_tensor, input_tensor_channel, output_tensor_channel, kernel_size = 3, do_quant = False):

        W = self.weights_msra(shape = [kernel_size, kernel_size, input_tensor_channel, output_tensor_channel])
        if do_quant:
            input_tensor = tf.fake_quant_with_min_max_args(input_tensor, min = -2.0, max = 2.0, num_bits = 4)
            W = tf.fake_quant_with_min_max_args(W, min = -1.0, max = 1.0, num_bits = 4)
        conv = tf.nn.conv2d(input_tensor, W, [1, 1, 1, 1], padding = "SAME")
        return conv + self.bias(shape = [output_tensor_channel])
    
    def conv_bn_relu(self, name, input_tensor, input_tensor_channel, output_tensor_channel, kernel_size = 3, do_quant = False):
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE): 
            conv = self.conv2d(input_tensor, input_tensor_channel, output_tensor_channel, kernel_size = kernel_size, do_quant = do_quant)
            # conv = tf.contrib.layers.batch_norm(conv, scale = True, is_training = True, updates_collections= None)
            conv_bn_scale = tf.get_variable(name = "bn_scale", shape = [output_tensor_channel], initializer = tf.ones_initializer, dtype = "float32")
            conv_bn_bias = tf.get_variable(name = "bn_bias", shape = [output_tensor_channel], initializer = tf.zeros_initializer, dtype = "float32")
            conv = self.bn(conv, conv_bn_scale, conv_bn_bias)
            return tf.nn.relu(conv)
            
        
    def conv_group(self, input_tensor, input_channel, output_channel, num, name, pooling_type = "MAX", final_kernel_size = 3, do_quant = False):
        conv_cnt = 0
        with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
            conv = self.conv_bn_relu(str(conv_cnt), input_tensor, input_channel, output_channel, do_quant = do_quant)
            conv_cnt += 1
            for i in range(1, num):
                if i == num - 1:
                    conv = self.conv_bn_relu(str(conv_cnt), conv, output_channel, output_channel, final_kernel_size, do_quant = do_quant)
                else:
                    conv = self.conv_bn_relu(str(conv_cnt), conv, output_channel, output_channel, do_quant = do_quant)
                conv_cnt += 1
            if pooling_type == "MAX":
                return tf.nn.max_pool(conv, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID") 
            else :
                return tf.nn.avg_pool(conv, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID") 

    def fc_bn_relu(self, input_tensor, input_channel, output_channel, name, do_quant = False):
        with tf.variable_scope(name):
            current = self.fc(input_tensor, input_channel, output_channel, name = "fc", do_quant = do_quant)
            current = tf.contrib.layers.batch_norm(current, scale = True, is_training = True, updates_collections = None)
            return tf.nn.relu(current)
             

    def fc(self, input_tensor, input_channel, output_channel, name, do_quant = False):
        with tf.variable_scope(name):
            weights = self.weights_xavier(shape = [input_channel, output_channel])
            if do_quant:
                input_tensor = tf.fake_quant_with_min_max_args(input_tensor, min = -2.0, max = 2.0, num_bits = 4) 
                weights = tf.fake_quant_with_min_max_args(weights, min = -2.0, max = 2.0, num_bits = 4)
            bias    = self.bias(shape = [output_channel]) 
            return tf.matmul(input_tensor, weights) + bias


    def build(self):
        
    # input
        self.X = tf.placeholder(dtype = "float32", shape = [None, 32, 32 ,3])
        self.Y = tf.placeholder(dtype = "float32", shape = [None, 100])
        self.lr= tf.placeholder(dtype = "float32")

    # net
        
        #conv1
        current = self.conv_group(name = "conv1", input_tensor = self.X, input_channel = 3, output_channel = 64, num = 2)
        #conv2
        current = self.conv_group(name = "conv2", input_tensor = current, input_channel = 64, output_channel = 128, num = 2, do_quant = True)
        #conv3
        current = self.conv_group(name = "conv3", input_tensor = current, input_channel = 128, output_channel = 256, num = 3, final_kernel_size = 1, do_quant = True)
        #conv 4
        current = self.conv_group(name = "conv4", input_tensor = current, input_channel = 256, output_channel = 512, num = 3, final_kernel_size = 1, do_quant = True)
        #conv 5
        current = self.conv_group(name = "conv5", input_tensor = current, input_channel = 512, output_channel = 512, num = 3, pooling_type = "AVG", final_kernel_size = 1, do_quant = True)
        #reshape for fc
        current = tf.reshape(current, [-1, 512])
        #fc_bn_relu1
        current = self.fc_bn_relu(name = "fc_bn_relu1", input_tensor = current, input_channel = 512, output_channel = 4096, do_quant = True)
        #fc_bn_relu2
        current = self.fc_bn_relu(name = "fc_bn_relu2", input_tensor = current, input_channel = 4096, output_channel = 4096, do_quant = True)
        #fc1
        current = self.fc(name = "fc1", input_tensor = current, input_channel = 4096, output_channel = 100)

        cross_entropy     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = current))
        l2_loss           = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss         = l2_loss * self.weight_decay + cross_entropy
        # self.train_step   = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        self.train_step   = tf.train.MomentumOptimizer(self.lr, 0.9,use_nesterov = True).minimize(self.loss)
        top1              = tf.equal(tf.argmax(current, 1), tf.argmax(self.Y, 1))
        self.top1_acc     = tf.reduce_mean(tf.cast(top1, "float"))
        top5              = tf.nn.in_top_k(predictions = current, targets = tf.argmax(self.Y, 1), k = 5) 
        self.top5_acc     = tf.reduce_mean(tf.cast(top5, "float"))


        # just for test
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("top1_acc", self.top1_acc)
        tf.summary.scalar("top5_acc", self.top5_acc)
        print(len(self.bn_group))


    def train(self, data_handle , saving_mode = False):
 #        config = tf.ConfigProto()
 #        config.gpu_options.allow_growth = True

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('.', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            lr = 0.01
            index = 0
            for epoch in range(1, 125 + 1):
                gen = data_handle.epoch_data_train(self.batch_size)
                cnt = 0
                if epoch == 50: lr /= 10
                if epoch == 75: lr /= 10
                if epoch == 100: lr /= 10
                for x, y in gen():
                    cnt += 1
                    index += 1
                    sess.run([self.train_step] + self.bn_group, feed_dict = {self.X: x,self.Y: y, self.lr: lr, self.is_training: True})

                    # ret = sess.run([merged_summary, self.train_step, self.loss, self.top1_acc, self.top5_acc] + self.bn_group, feed_dict = {self.X: x,self.Y: y, self.lr: lr, self.is_training: True})
                    if cnt % 5 == 0:
                        
                        s, train_loss, train_acc, train_top5_acc = sess.run([merged_summary, self.loss, self.top1_acc, self.top5_acc], feed_dict = {self.X: x,self.Y: y, self.lr: lr, self.is_training: True})
                        writer.add_summary(s, index)
                        print("train epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, train_loss, train_acc, train_top5_acc)) 
                if epoch % 5 == 0:
                    x, y = data_handle.data_test()
                    test_loss, test_acc, test_top5_acc = sess.run([self.loss, self.top1_acc, self.top5_acc], feed_dict = {self.X: x, self.Y: y, self.is_training: False})
                    print("test epoch = {:d}, loss = {:f}, top_1_acc = {:f}, top_5_acc = {:f}".format(epoch, test_loss, test_acc, test_top5_acc))
        # just for test

        writer.close()



        
    def test(self, data_manager):
        pass

    def save_graph(self, path):
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(path, tf.get_default_graph())
        writer.close()

        

if __name__ == "__main__":
#     a = tf.Graph()
    # b = tf.()
    # if a is b:
        # print("same")
    # else:
        # print("not same")
    # a.as_default()
    # if a is tf.get_default_graph():
        # print("is default_graph")
    # else:
        # print("not defautl")
    
    data_handle = data_manager.Data_Manager("cifar-100-python/train", "cifar-100-python/test")
    # data_handle = data_manager.Data_Manager("train", "test")
    vgg = Vgg()
    vgg.build()
    vgg.train(data_handle)
    # vgg.save_graph(".")
    
  #   for elem in tf.trainable_variables():
        # print(elem)

    # for n in tf.get_default_graph().as_graph_def().node:
  #       print(n.name)

    # can find 

        








