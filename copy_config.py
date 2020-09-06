import argparse
import inspect
import os
import pickle
from shutil import copyfile

# called from training script to parse arguments and copy experiment config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select an experiment name and override values in its config')

    parser.add_argument("--exp", type=str, help="Name of experiment", required=True)
    parser.add_argument("--test_mode", help="Perform test run", default=False, action='store_true')

    try:
        from configs import Config

        # add all members of current config as possible arguments
        for name, value in inspect.getmembers(Config):
            if type(value) is bool:
                parser.add_argument(f"--{name}", dest=name, default=None, action='store_true')
                parser.add_argument(f"--no-{name}", dest=name, action='store_false')
            else:
                if type(value) in {int, float}:
                    t = type(value)
                else:
                    t = str
                parser.add_argument(f"--{name}", type=t, default=None)

        parsed_args, _ = parser.parse_known_args()
    except Exception as ex:
        print(f"Current config file failed to load with message: {ex} \n"
              f"No additional commandline parameters available \n")
        parsed_args, _ = parser.parse_known_args()

    copyfile(os.path.join('configs', 'experiments', parsed_args.exp + '.py'),
             os.path.join('configs', '__init__.py'))

    # save parsed arguments for later use in training script
    with open('__parsed_args.pkl', 'wb') as file:
        pickle.dump(parsed_args, file)
