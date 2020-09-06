from collections import defaultdict

dd = lambda x: defaultdict(lambda: x)


class Contextuable(type):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SpecifiableConfig(type):
    def __getitem__(self, item):
        class ContextualConfig(type(self), metaclass=Contextuable):
            pass

        for attr in dir(self):
            value = getattr(self, attr)
            if (isinstance(value, dict) and item in value) or isinstance(value, defaultdict):
                setattr(ContextualConfig, attr, value[item])
            elif not attr.startswith('__'):
                setattr(ContextualConfig, attr, value)
        return ContextualConfig


class BaseConfig(metaclass=SpecifiableConfig):
    pass


class SystemConfig(BaseConfig):
    logs_path = 'logs'
    logs_file = 'run.log'
    checkpoints_folder = 'checkpoints'
    optimizers_file = 'optimizers.pth'
    amp_file = 'amp.pth'

    date_format = '%y-%m-%d'
    time_format = '%H-%M-%S'
    log_format = '%(levelname)s - %(module)s:%(funcName)s:%(lineno)s - %(asctime)s - %(message)s'
    log_time_format = date_format + ':' + time_format

    log_tensorboard = True
    log_terminal = False

    n_cores = 12
