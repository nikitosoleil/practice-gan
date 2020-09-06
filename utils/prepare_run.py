import os
import pickle
import subprocess
import sys
import torch


def prepare_run():
    # copy experiment config in different process to prevent import issues
    subprocess.run(['python', 'copy_config.py'] + sys.argv[1:])

    # read parsed arguments from temporal file and delete it
    with open('__parsed_args.pkl', 'rb') as file:
        parsed_args = pickle.load(file)
    os.remove('__parsed_args.pkl')

    from configs import Config

    # update config with new arguments
    for name, value in parsed_args.__dict__.items():
        if value is not None:
            setattr(Config, name, value)
    getattr(Config, 'init', lambda: None)()

    # for model in Config.models:
    #     Config.device[model] = torch.device(Config.device[model])
