import argparse
import subprocess

from randomized_hyperparam_search import RandomizedSearch, ga_reader_parameter_space

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_times', help='Number of times to choose random parameters and train the model with them.',
                        type=int, default=10)
    search_args = parser.parse_args()

    random_search = RandomizedSearch(ga_reader_parameter_space)
    for n in range(search_args.n_times):
        print("Run {}...".format(n))
        params = random_search.sample()
        print(params)
        command = " ".join(["THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python3", "run.py"] +
                           ["--dataset clicr_plain --mode 1 --data_path /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/bmj_case_reports_data/dataset_plain/ent/gareader/ --experiments_path /mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/ga-reader/experiments/"] +
                           ["--{} {}".format(k, v) for k, v in params.items()]
                           )
        subprocess.call(command, shell=True)
