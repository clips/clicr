import argparse
import subprocess

from utils.utils import get_file_list


if __name__ == "__main__":
    for i in get_file_list("/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/ga-reader/experiments/", identifiers=["best_model.p"], all_levels=True):
        topdir, model_dir, best_model_file = i.split("/")[-3], i.split("/")[-2],  i.split("/")[-1]
        model_params = model_dir.split("_")
        params = {}
        params["nhid"] = model_params[1][len("nhid"):]
        params["dropout"] = model_params[3][len("dropout"):]
        params["use_feat"] = model_params[8][len("use-feat"):]
        print(params)      
        command = " ".join(["THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python", "run.py"] +
                           ["--dataset clicr --mode 2"] +
                           ["--{} {}".format(k, v) for k, v in params.items()])
        print(command)
        subprocess.call(command, shell=True)
