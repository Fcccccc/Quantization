# first, change the nchw or nhwc
# second,  shuffle the input and normalize
# then, augment images cropping, padding, flippling(mirror)
# batch generate images..



import numpy as np
import pickle
import keras
import sklearn
import math


class Data_Manager:

    def __init__(self, train_file, test_file):
        self.train_data, self.train_labels = self.generateXY(train_file)
        self.test_data, self.test_labels = self.generateXY(test_file)

    # just for cifar100
    def unpickle(self, file_name):
        with open(file_name, 'rb') as fd:
            dic = pickle.load(fd, encoding = 'bytes')
            dic[b'data'] = dic[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
            return dic

    def get_mean_and_std(self, images):
        means = []
        stds  = []
        for ch in range(images.shape[-1]):
            means.append(np.mean(images[:, :, :, ch]))
            stds.append(np.std(images[:, :, :, ch]))
        return means, stds;
    def normalize_images(self, images):
        means, stds = self.get_mean_and_std(images)
        for ch in range(images.shape[-1]):
            images[:, :, :, ch] = ((images[:, :, :, ch] - means[ch]) / stds[ch])
        return images

    def augment_images(self, data):
        shape = data.shape
        pad   = 4
        for i in range(data.shape[0]):
            new_shape = [shape[1] + pad * 2, shape[2] + pad * 2, shape[3]]
            zeros = np.zeros(new_shape)
            zeros[pad:shape[1] + pad, pad:shape[2] + pad, :] = data[i]
            x = np.random.randint(0, pad * 2)
            y = np.random.randint(0, pad * 2)
            data[i] = zeros[x:shape[1] + x, y:shape[2] + y, :]
            if np.random.randint(2):
                data[i] = data[i][:, ::-1, :] 
        return data

    def shuffle(self):
        index = sklearn.utils.shuffle([i for i in range(self.train_data.shape[0])])
        return self.train_data[index], self.train_labels[index]

    def epoch_data_train(self, batch):
        _new_data, new_labels = self.shuffle()
        def func(_batch = batch):
            new_data = self.augment_images(_new_data)
            num = new_data.shape[0]
            for i in range((num + batch - 1) // batch):
                l = i * batch
                r = min((i + 1) * batch, new_data.shape[0])
                yield new_data[l: r], new_labels[l: r]
        return func
    def data_test(self):
        return self.test_data, self.test_labels

    # first shuffle
    def generateXY(self, filename):
        dic = self.unpickle(filename)
        X   = dic[b'data']
        Y   = dic[b'fine_labels']
        X   = np.asarray(X)
        Y   = np.asarray(Y)
        Y   = keras.utils.to_categorical(Y, 100)
        index = [i for i in range(Y.shape[0])]
        index = sklearn.utils.shuffle(index) 
        X   = X.astype(np.float32)
        Y   = Y.astype(np.float32)
        return self.normalize_images(X[index]), Y[index]


if __name__ == "__main__":
    data_manager = Data_Manager(train_file = "~/Draft/cifar-100-python/train", test_file = "~/Draft/cifar-100-python/test")
    # data_manager = Data_Manager(train_file = "train", test_file = "test")
    print(data_manager.train_data)
    print(data_manager.train_labels)
    gen = data_manager.epoch_data_train(128)
    for X,Y in gen():
        print(X)
        print(Y)
