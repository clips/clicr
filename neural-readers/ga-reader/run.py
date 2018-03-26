import os

#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import sys

import train
import test
import argparse
import numpy as np
import random

from config import get_params

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', dest='mode', type=int, default=0,
                    help='run mode - (0-train+test, 1-train only, 2-test only, 3-val only)')
parser.add_argument('--experiments_path', default='experiments/')
parser.add_argument('--data_path')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=3,
                    help='Number of reader layers')
parser.add_argument('--nhidden', dest='nhidden', type=int, default=128,
                    help='Number of hidden units')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='Dropout rate')
parser.add_argument('--use_feat', dest='use_feat', type=int, default=0,
                    help='use indicator feature (0-no, 1-yes)')
parser.add_argument('--dataset', dest='dataset', type=str, default='wdw',
                    help='Dataset - (cnn || dailymail || cbtcn || cbtne || wdw || clicr || clicr_plain || clicr_novice)')
parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='Seed for different experiments with same settings')
parser.add_argument('--gating_fn', dest='gating_fn', type=str, default='T.mul',
                    help='Gating function (T.mul || Tsum || Tconcat)')
parser.add_argument('--ent_setup', dest='ent_setup', type=str, default='ent-anonym',
                    help='Setup (ent-anonym || ent || no-ent)')
args = parser.parse_args()
cmd = vars(args)
params = get_params(cmd['dataset'])
if args.data_path is None and params["data_path"] is not None:
    args.__setattr__("data_path", params["data_path"])
params.update(cmd)
np.random.seed(params['seed'])
random.seed(params['seed'])

# save directory
w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'

setup_name = '_stp%s' % params['ent_setup'] if args.dataset.startswith("clicr") else ""
save_path = (args.experiments_path +
             #'/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Tools/ga-reader/experiments/' +
             params['dataset'].split('/')[0] +
             '_nhid%d' % params['nhidden'] + '_nlayers%d' % params['nlayers'] +
             '_dropout%.1f' % params['dropout'] + '_%s' % w2v_filename + '_chardim%d' % params['char_dim'] +
             '_train%d' % params['train_emb'] +
             '_seed%d' % params['seed'] + '_use-feat%d' % params['use_feat'] +
             '_gf%s' % params['gating_fn'] +
              setup_name +
             '/')
if not os.path.exists(save_path): os.makedirs(save_path)
print("Writing to: {}".format(save_path))
with open(save_path + "command_runmode{}.log".format(args.mode), "w") as fh_out:
    fh_out.write(" ".join(sys.argv))
    for k,v in params.items():
        fh_out.write("{}: {}\n".format(k, v))

# train
if params['mode'] < 2:
    train.main(save_path, params)

# test
if params['mode'] == 0 or params['mode'] == 2:
    test.main(save_path, params)
elif params['mode'] == 3:
    test.main(save_path, params, mode='validation')
