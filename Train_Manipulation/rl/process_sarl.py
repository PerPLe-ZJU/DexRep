
from model.ppo import PPO

def process_sarl(args, env, cfg_train, logdir):
    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]

    if learn_cfg['resume_model'] != '':
        args.resume_model = learn_cfg['resume_model']
    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if args.resume_model != "":
        # is_testing = True
        chkpt_path = args.resume_model

    """Set up the algo system for training or inferencing."""
    # if args.algo.upper()=='PPO':
    model = eval(args.algo.upper())(vec_env=env,
                                    cfg_train=cfg_train,
                                    cfg_env=args.task_envs,
                                    sampler=learn_cfg.get("sampler", 'sequential'),
                                    log_dir=logdir,
                                    is_testing=is_testing,
                                    print_log=learn_cfg["print_log"],
                                    device=args.rl_device
                                    )

    # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
    if is_testing and args.resume_model != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif args.resume_model != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path)

    return model
