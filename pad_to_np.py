#coding:utf-8
###################################################
# File Name: pad_to_np.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年04月02日 星期二 17时12分25秒
#=============================================================
import paddle.fluid as fluid
import joblib
from model.ernie import ErnieConfig
from utils.init import init_checkpoint, init_pretraining_params
from finetune.classifier import create_model


import numpy as np
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--init_checkpoint", default='/root/zhaomeng/baidu_ERNIE/LARK/ERNIE/params', type=str, help=".")
parser.add_argument("--ernie_config_path", default='/root/zhaomeng/baidu_ERNIE/LARK/ERNIE/config/ernie_config.json', type=str, help=".")
parser.add_argument("--max_seq_len", default=128, type=int, help=".")
parser.add_argument("--num_labels", default=2, type=int, help=".")
parser.add_argument("--use_fp16", type=bool, default=False, help="Whether to use fp16 mixed precision training.")


args = parser.parse_args()





if __name__ == '__main__':
    if not args.init_checkpoint:
        raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")

    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    test_program = fluid.Program()

    with fluid.program_guard(test_program, startup_prog):
        with fluid.unique_name.guard():
            _, _ = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config)
    
    exe.run(startup_prog)
    

    init_pretraining_params(
                    exe,   
                    args.init_checkpoint,
                    main_program=test_program,
                    #main_program=startup_prog,
                    use_fp16=args.use_fp16)

    name2params = {}
    prefix = args.init_checkpoint
    for var in startup_prog.list_vars():
        path = os.path.join(prefix, var.name)
        if os.path.exists(path):
            cur_tensor = fluid.global_scope().find_var(var.name).get_tensor()
            print(var.name, np.array(cur_tensor).shape)
            name2params[var.name] = np.array(cur_tensor)

    joblib.dump(name2params, 'params.dict') 

