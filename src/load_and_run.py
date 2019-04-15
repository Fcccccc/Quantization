#!/usr/bin/env python3
import tensorflow as tf
import sys
import os
import importlib
from Utils import data_manager, cnnUtils


if __name__ == "__main__":
    model_name = sys.argv[1]

    model = importlib.import_module("model." + model_name)
    model_cls = getattr(model, model_name)
    model_obj = model_cls([None, 32, 32, 3], [None, 100])
    model_obj.build()
    data_handle = data_manager.Data_Manager("/Users/zhangfucheng/data/cifar-100-python/train", "/Users/zhangfucheng/data/cifar-100-python/test")
    # data_handle = data_manager.Data_Manager("/home/zhangfucheng/Draft/train", "/home/zhangfucheng/Draft/test")
    cnnUtils.train(model_obj, 128, data_handle)
 
