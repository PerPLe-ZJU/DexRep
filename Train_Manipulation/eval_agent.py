import os
import sys

from tqdm import tqdm

from utils.logger import DataLog
import os
from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from rl.process_sarl import process_sarl
import torch
import numpy as np
def main():

    # Set up python env
    set_np_formatting()
    args = get_args()
    set_seed(args.models['seed'], args.models['torch_deterministic'])

    # Construct task
    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    # args.resume_model
    logger = DataLog()
    assert os.path.isdir(args.logger_dir)

    eval_seed = args.eval_seed

    model_paths = []
    if args.algo == 'dagger':
        model_path = args.logger_dir
    else:
        for s1 in eval_seed:
            model_paths.append(os.path.join(args.resume_model, s1, 'checkpoint'))

    for mp in model_paths:
        for model_iter in args.eval_iter:
            # args.resume_model = os.path.join(model_path, f'model_{model_iter}.pt')
            args.resume_model = os.path.join(mp, f'model_{model_iter}.pt')
            # logger.log_kv('model', f'model_{i}')
            # set up policy
            sarl = process_sarl(args, env, args.models, args.logger_dir)
            sarl.eval(logger, max_trajs=10)
            # logger.log_kv('reward', reward)
            # logger.log_kv('success_rate', success)
            del sarl
            #
            save_path = os.path.dirname(mp)
            print(save_path)
            logger.save_log(save_path, 'evaluation')

    sys.exit(0)


if __name__ == '__main__':
    main()