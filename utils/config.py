# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def warn_algorithm_name():
    raise Exception(
                "Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args, use_rlg_config=False):
    if args.task == "ShadowHandGrasp":
        return os.path.join(args.logdir, "shadow_hand_grasp/{}/{}".format(args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_grasp.yaml"
    elif args.task == "ShadowHandRandomLoadVision":
        return os.path.join(args.logdir, "shadow_hand_random_load_vision/{}/{}".format(args.algo,args.algo)), "cfg/{}/config.yaml".format(args.algo), "cfg/shadow_hand_random_load_vision.yaml"
    else:
        warn_task_name()



def load_cfg(args, use_rlg_config=False):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    if use_rlg_config:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["params"]["torch_deterministic"] = True

        exp_name = cfg_train["params"]["config"]['name']

        if args.experiment != 'Base':
            if args.metadata:
                exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

                if cfg["task"]["randomize"]:
                    exp_name += "_DR"
            else:
                exp_name = args.experiment

        # Override config name
        cfg_train["params"]["config"]['name'] = exp_name

        if args.resume > 0:
            cfg_train["params"]["load_checkpoint"] = True

        if args.checkpoint != "Base":
            cfg_train["params"]["load_path"] = args.checkpoint

        # Set maximum number of training iterations (epochs)
        if args.max_iterations > 0:
            cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

        cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

        seed = cfg_train["params"].get("seed", -1)
        if args.seed is not None:
            seed = args.seed
        cfg["seed"] = seed
        cfg_train["params"]["seed"] = seed

        cfg["args"] = args
    else:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["torch_deterministic"] = True

        # Override seed if passed on the command line
        if args.seed is not None:
            cfg_train["seed"] = args.seed

        log_id = args.logdir
        if args.experiment != 'Base':
            if args.metadata:
                log_id = args.logdir + "_{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])
                if cfg["task"]["randomize"]:
                    log_id += "_DR"
            else:
                log_id = args.logdir + "_{}".format(args.experiment)

        logdir = os.path.realpath(log_id)
        # os.makedirs(logdir, exist_ok=True)
    
    ocd_tag = args.ocd_tag

    if args.backbone_type != "":
        cfg_train["policy"]["backbone_type"] = args.backbone_type
    else:
        cfg_train["policy"]["backbone_type"] = None
        
    
    cfg_train["policy"]["freeze_backbone"] = args.freeze_backbone
        
    if len(ocd_tag) > 0:
        ocd_group_file = 'eval_results/object_code_dict_groups.yaml'
        with open(ocd_group_file, 'r') as f:
            ocd_groups = yaml.safe_load(f)
        ocd = {}
        for tag in ocd_groups['run_tags'][ocd_tag]:
            ocd.update(ocd_groups['ocd_groups'][tag])
        cfg['env']['object_code_dict'] = ocd

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


import os
import argparse
import yaml
from utils.util import pretty
from utils.util import DotDict
import torch


def get_args():
    parser = argparse.ArgumentParser(description='manipulation tasks')

    parser.add_argument('--task', type=str, required=True, help='manipulated task for an agent')
    parser.add_argument('--algo', type=str, default='PPO', help='training algorithm')
    parser.add_argument('--resume_model', type=str, default='', help='Choose a model dir')
    parser.add_argument('--device', type=str, default='cpu', help='device for training')
    parser.add_argument('--env_vis', action='store_true', help='visualize envs or not')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--seed',required=True,  type=int, help='test')
    parser.add_argument('--allegro', action='store_true', help='allegro hand')

    args = parser.parse_args()

    if args.allegro:
        args.task = args.task + '_allegro_hand'
    assert args.task + '.yaml' in os.listdir('./config/task_env'), f"task is not defined, please choose one from {os.listdir('./config/task_env')}(except '.yaml')"
    with open('./config/task_env/'+args.task+'.yaml') as f:
        args.task_envs = yaml.load(f, Loader=yaml.FullLoader)
    args.task_envs['seed'] = args.seed
    # assert len(args.task_envs['env_names']) == args.task_envs['env_num']
    assert len(args.task_envs['env_names']) == args.task_envs['env_num']
    args.task_envs['env_path'] = os.path.join(os.getcwd(), args.task_envs['env_path'])
    # load model params
    with open(f'./config/algos/ppo/{args.task}.yaml') as f:
        args.models = yaml.load(f, Loader=yaml.FullLoader)
    if args.test:
        args.models['learn']['test'] = args.test
        args.models["learn"]["n_per_env"] = 1
        if args.env_vis:
            args.task_envs['img_w'] = 1080
            args.task_envs['img_h'] = 1080
        # args.task_envs['ep_length'] = 500
    # logger dir
    curr_dir = os.getcwd()
    args.logger_dir = os.path.join(curr_dir, 'runs', args.task, 'seed'+str(args.task_envs['seed']))
    os.makedirs(args.logger_dir, exist_ok=True)

    # args.task_envs, args.models = DotDict(args.task_envs), DotDict(args.models)
    #str->torch.device
    # args.device = torch.device(args.device)
    # visualize envs
    args.task_envs['visualize'] = args.env_vis
    del args.env_vis

    # args.train_params = args.task_envs['train_params']
    # del args.task_envs['train_params']
    if not args.models['learn']['test']:
        config_params = vars(args)
        # save all config
        with open(os.path.join(args.logger_dir, 'all_config.yaml'), 'w') as f:
            f.write(yaml.dump(config_params))

        print(pretty(config_params))
    return args
