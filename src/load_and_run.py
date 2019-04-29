#!/usr/bin/env python3
# import tensorflow as tf
# import sys
# import os
# import importlib
import argparse
import os
from src.Utils import data_manager, nnUtils
# from src.Utils import data_manager, nnUtils


hyperparams = {
        "model_name":None,
        "input_shape":[None, 32, 32, 3],
        "output_shape":[None, 100],
        "train_batchsize":128,
        # "train_epoch":5,
        "train_epoch":125,
        "train_lr":0.01,
        "train_lr_reduce":[50, 75, 100],
        "weights_decay":0.0005,
        "enable_quant":False,
        "quant_bits":False,
        "enable_prune":False,
        "prune_batchsize":128,
        # "prune_epoch":1,
        "prune_epoch":50,
        "prune_lr":0.001,
        "prune_lr_reduce":[25],
        "begin_pruning_step":0,
        "end_pruning_step":-1,
        "pruning_frequency":5,
        "target_sparsity":False,
        "enable_dump_weights":False}


def do_train(model_obj, hyperparams):
    if not os.path.exists("../var"):
        os.mkdir("../var")
    os.chdir("../var")
    data_handle = data_manager.Data_Manager("/Users/zhangfucheng/data/cifar-100-python/train", "/Users/zhangfucheng/data/cifar-100-python/train")
    # data_handle = data_manager.Data_Manager("/home/jason/Draft/cifar-100-python/train", "/home/jason/Draft/cifar-100-python/test")
    # data_handle = data_manager.Data_Manager("/home/zhangfucheng/Draft/train", "/home/zhangfucheng/Draft/test")
    trainer = nnUtils.Trainer(model_obj, data_handle, hyperparams)
    print(type(trainer))
    trainer.do_train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quant_bits", metavar = "bits", help = "num_bits used to quantize the model", type = int, nargs = '?', default = False)
    parser.add_argument("model_name", help = "model_name")
    parser.add_argument("-p", "--prune", metavar = "sparsity", dest = "prune_sparsity", help = "enable_prune", type = float, default = False, nargs = '?')
    parser.add_argument("-d", "--dump_weights", help = "enable_dumpe_weights", action = "store_true", default = False)
    parser.add_argument("-s", "--size", metavar = "weights_name", help = "print model size", nargs = '?', default = False)
    args = parser.parse_args()
    print(args)

    if args.quant_bits == None:
        args.quant_bits = 8
    if args.prune_sparsity == None:
        args.prune_sparsity = 0.6

    if args.quant_bits:
        hyperparams['enable_quant'] = True
        hyperparams['quant_bits']   = args.quant_bits
    if args.prune_sparsity:
        hyperparams['enable_prune'] = True
        hyperparams['target_sparsity'] = args.prune_sparsity
    if args.dump_weights:
        hyperparams['enable_dump_weights'] = True
    hyperparams['model_name'] = args.model_name

    import tensorflow as tf
    import sys
    import importlib
    
    if hyperparams['enable_prune'] and hyperparams['enable_quant']:
        model = importlib.import_module("quant_and_prune_model." + args.model_name)
    elif hyperparams['enable_prune']:
        model = importlib.import_module("prune_model." + args.model_name)
    elif hyperparams['enable_quant']:
        model = importlib.import_module("quant_model." + args.model_name)
    else:
        model = importlib.import_module("normal_model." + args.model_name)

    model_cls = getattr(model, args.model_name)
    model_obj = model_cls(hyperparams)
    model_obj.build()
    if args.size == None or args.size:
        cnt_bits = int(nnUtils.model_size(args.size))
        cnt_bits *= 32
        print("{:f} bits, {:f} B, {:f} KB, {:f} MB".format(cnt_bits, cnt_bits / 8, cnt_bits / (8 * 1024), cnt_bits / (8 * 1024 * 1024)))

    else:
        do_train(model_obj, hyperparams)



main()

# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-q", "--quant_bits", help = "num_bits used to quantize the model", type = int, nargs = '?', default = False)
    # parser.add_argument("-m", "--model_name", help = "model_name")
    # parser.add_argument("-p", "--prune_on", dest = "enable_prune", help = "enable_prune", action = "store_true")
    # args = parser.parse_args()
    # print(args)
    # if args.quant_bits == None:
        # args.quant_bits = 8

    # print(args.quant_bits)
    # print(args.model_name)
    # print(args.enable_prune)
#     # model_name = sys.argv[1]
    # enable_quant = False
    # enable_prune = False
    # if model_name[-5:] == "quant":
        # enable_quant = True
        # quant_bits = int(sys.argv[2])
    # elif model_name[-5:] == "prune":
        # enable_prune = True
    
#     if enable_prune:
        # model = importlib.import_module("prune_model." + model_name)
    # else:
        # model = importlib.import_module("model." + model_name)
    # model_cls = getattr(model, model_name)
    # if enable_quant:
        # model_obj = model_cls([None, 32, 32, 3], [None, 100], quant_bits = quant_bits)
    # else:
        # model_obj = model_cls([None, 32, 32, 3], [None, 100])
    # model_obj.build()
    # data_handle = data_manager.Data_Manager("/Users/zhangfucheng/data/cifar-100-python/train", "/Users/zhangfucheng/data/cifar-100-python/test")
    # # data_handle = data_manager.Data_Manager("/home/zhangfucheng/Draft/train", "/home/zhangfucheng/Draft/test")
    # nnUtils.train(model_obj, 128, data_handle, model_name, enable_prune)
 

