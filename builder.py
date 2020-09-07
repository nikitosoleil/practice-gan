import logging
import os
import random
from shutil import copyfile
import time

import torch
from apex import amp
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from configs import Config, Components
from utils.checkpointer import Checkpointer
from utils.writers import TBWriter, TerminalWriter, CombinedWriter
from utils.reporter import Reporter


class Builder:
    def __init__(self):
        self.base_logs_path = os.path.join(Config.logs_path, Config.exp)

        self.start_date = None
        self.start_time = None

        self.logs_path = None
        self.logs_file_path = None

        self.log_handlers = []
        self.writers = []
        self.tb_writer = None
        self.reporter = None

        self.dataset = None

        self.restoration_path = None
        self.checkpointer = None

        self.models = {}
        self.optimizers = {}

        self.optimizers_state_dicts = None
        self.amp_state_dict = None

        if Config.seed is not None:
            random.seed(Config.seed)
            np.random.seed(Config.seed)
            torch.manual_seed(Config.seed)

    def init_log_folder(self):
        self.start_date = time.strftime(Config.date_format)
        self.start_time = time.strftime(Config.time_format)
        self.logs_path = os.path.join(self.base_logs_path, self.start_date, self.start_time)
        os.makedirs(self.logs_path)

        copyfile(os.path.join('configs', 'experiments', Config.exp + '.py'),
                 os.path.join(self.logs_path, '__init__.py'))

    def add_file_log_handler(self):
        self.logs_file_path = os.path.join(self.logs_path, Config.logs_file)
        self.log_handlers.append(logging.FileHandler(self.logs_file_path))

    def add_stream_log_handler(self):
        self.log_handlers.append(logging.StreamHandler())

    def create_logger(self):
        logging.basicConfig(level=logging.INFO, format=Config.log_format, datefmt=Config.log_time_format,
                            handlers=self.log_handlers)
        logging.info(f'Log file initialized at {self.logs_file_path}')

    def create_checkpointer(self):
        checkpoints_path = os.path.join(self.logs_path, Config.checkpoints_folder)
        self.checkpointer = Checkpointer(checkpoints_path)
        logging.info(f'Checkpointer initialized at {checkpoints_path}')

    def create_tensorboard_writer(self):
        self.tb_writer = TBWriter(SummaryWriter(self.logs_path, flush_secs=10))
        self.writers.append(self.tb_writer)
        logging.info(f'Tensorboard writer initialized at {self.logs_path}')

    def create_terminal_writer(self):
        self.writers.append(TerminalWriter())
        logging.info(f'Terminal writer initialized')

    def create_reporter(self):
        self.reporter = Reporter(CombinedWriter(self.writers))
        logging.info(f'Reporter initialized')

    def create_dataset(self):
        self.dataset = Components.dataset()
        logging.info(f'Dataset created')

    def find_restoration_path(self):
        if Config.restore_from in {'-1', -1}:
            candidates = sorted(os.path.join(date, time) for date in os.listdir(self.base_logs_path)
                                for time in os.listdir(os.path.join(self.base_logs_path, date))
                                if len(os.listdir(os.path.join(self.base_logs_path, date, time,
                                                               Config.checkpoints_folder))) != 0)
            if len(candidates) == 0:
                raise ValueError('No available checkpoints to resto re from')
            restore_from = candidates[-1]
        else:
            restore_from = Config.restore_from

        checkpoints_path = os.path.join(self.base_logs_path, restore_from, Config.checkpoints_folder)

        time = int(Config.restore_time) if Config.restore_time not in {'-1', -1} else \
            max([int(file.split('.')[0].split('_')[1]) for file in os.listdir(checkpoints_path)])
        self.restoration_path = os.path.join(checkpoints_path, f'time_{time}')

        logging.info(f'Restoration path found to be {self.restoration_path}')

    def create_models(self):
        for name in Config.models:
            self.models[name] = Components.models[name](name, Components.networks[name],
                                                        self.restoration_path if name in Config.trainable_models else None)
        logging.info('Models created')

    def create_optimizers(self):
        for name in Config.models:
            params = self.models[name].network.parameters()
            self.optimizers[name] = Components.optimizers[name](
                params, lr=Config.learning_rate[name])

        logging.info('Optimizers created')

    def apply_mixed_precision(self):
        applicable_models = [model for model in Config.models if Config.device[model] != 'cpu']
        networks, optimizers = zip(*[(self.models[model].network, self.optimizers[model]) for model in applicable_models])

        networks, optimizers = amp.initialize(list(networks), list(optimizers), opt_level=Config.opt_level)

        for m, network, optimizer in zip(applicable_models, networks, optimizers):
            self.models[m].network = network
            self.optimizers[m] = optimizer
        logging.info('Amp initialized')

    def restore_optimizers(self):
        optimizers_state_dicts = torch.load(os.path.join(self.restoration_path, Config.optimizers_file))
        for model in Config.trainable_models:
            self.optimizers[model].load_state_dict(optimizers_state_dicts[model])
        logging.info('Optimizers restored')

    def restore_amp(self):
        amp_path = os.path.join(self.restoration_path, Config.amp_file)
        if os.path.exists(amp_path):
            amp_state_dict = torch.load(amp_path)
            amp.load_state_dict(amp_state_dict)
            logging.info('Amp restored')
        else:
            logging.warning(f'Amp directory {amp_path} not found')
