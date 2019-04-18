#!/usr/bin/env python3
import tensorflow as tf
import sys
import os
import importlib
from src.Utils import data_manager, nnUtils


if __name__ == "__main__":
    model_name = sys.argv[1]
    enable_quant = False
    enable_prune = False
    if model_name[-5:] == "quant":
        enable_quant = True
        quant_bits = int(sys.argv[2])
    elif model_name[-5:] == "prune":
        enable_prune = True
    
    if enable_prune:
        model = importlib.import_module("prune_model." + model_name)
    else:
        model = importlib.import_module("model." + model_name)
    model_cls = getattr(model, model_name)
    if enable_quant:
        model_obj = model_cls([None, 32, 32, 3], [None, 100], quant_bits = quant_bits)
    else:
        model_obj = model_cls([None, 32, 32, 3], [None, 100])
    model_obj.build()
    data_handle = data_manager.Data_Manager("/Users/zhangfucheng/data/cifar-100-python/train", "/Users/zhangfucheng/data/cifar-100-python/test")
    # data_handle = data_manager.Data_Manager("/home/zhangfucheng/Draft/train", "/home/zhangfucheng/Draft/test")
    nnUtils.train(model_obj, 128, data_handle, model_name, enable_prune)
 

