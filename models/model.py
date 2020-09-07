import os
from typing import Type

import torch

from configs import Config


class BaseModel:
    model_file = 'pytorch_model.bin'

    def __init__(self, name, network_cls, restore_path=None):
        self.name = name
        self._load(network_cls, restore_path)
        self.config: Type[Config] = Config[self.name]
        self.network = self.network.to(self.config.device)

    def _get_component_path(self, base_path, component=None):
        component_path = os.path.join(base_path, self.name)
        if component is not None:
            component_path = os.path.join(component_path, component)
        return component_path

    def _get_path(self, base_path, component=None):
        component_path = self._get_component_path(base_path, component)
        if not os.path.exists(component_path):
            raise FileNotFoundError(f'Path {component_path} for TextModel {component} not found')
        return component_path

    def _create_path(self, base_path, component=None):
        component_path = self._get_component_path(base_path, component)
        os.makedirs(component_path)
        return component_path

    def _load(self, network_cls, restore_path):
        self.network = network_cls()

        if restore_path is not None:
            state_dict_path = os.path.join(self._get_path(restore_path), self.model_file)
            state_dict = torch.load(state_dict_path)
            self.network.load_state_dict(state_dict)

    def save(self, base_path):
        state_dict_path = os.path.join(self._create_path(base_path), self.model_file)
        state_dict = self.network.state_dict()
        torch.save(state_dict, state_dict_path)

    def flush(self):
        raise NotImplementedError

    def metric_run(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)[:2]
        return output

    def train(self, *args, **kwargs):
        self.network.train()
        return self.metric_run(*args, **kwargs)

    def validate(self, *args, **kwargs):
        self.network.eval()
        with torch.no_grad():
            return self.metric_run(*args, **kwargs)

    def forward(self, *args, **kwargs):
        def f(arg):
            return arg.to(self.config.device) if isinstance(arg, torch.Tensor) else arg

        nargs = [f(arg) for arg in args]
        nkwargs = {key: f(arg) for key, arg in kwargs.items()}
        outputs = self.network(*nargs, **nkwargs)
        return outputs

    def forward_batched(self, *args, postprocessing=None, **kwargs):
        bs, batches = None, None
        mbs = self.config.max_batch_size

        for key, arg in list(zip([None] * len(args), args)) + \
                        list(kwargs.items()):
            if isinstance(arg, torch.Tensor):
                assert arg.dim() > 0
                if bs is None:
                    bs = arg.shape[0]
                    batches = [[] for _ in range((bs + mbs - 1) // mbs)]
                else:
                    assert arg.shape[0] == bs
                batches_arg = torch.split(arg, self.config.max_batch_size)
                for batch, batch_arg in zip(batches, batches_arg):
                    batch.append((key, batch_arg))
            else:
                for batch in batches:
                    batch.append((key, arg))

        output_not_tuple = False
        results = []
        for batch in batches:
            nargs, nkwargs = [], {}
            for key, arg in batch:
                if isinstance(arg, torch.Tensor):
                    arg = arg.to(self.config.device)
                if key is None:
                    nargs.append(arg)
                else:
                    nkwargs[key] = arg
            outputs = self.network(*nargs, **nkwargs)

            if postprocessing is not None:
                outputs = postprocessing(*outputs)

            if not isinstance(outputs, tuple):
                output_not_tuple = True
                outputs = (outputs,)

            results.append(outputs)

        result = tuple()
        for batches in zip(*results):
            batches_processed = []
            for batch in batches:
                if isinstance(batch, torch.Tensor):
                    batch = batch if batch.dim() > 0 else batch.unsqueeze(0)
                else:
                    raise ValueError(f'Only torch.Tensor model outputs currently supported, got {type(batch)}')
                batches_processed.append(batch)
            batches_concatenated = torch.cat(batches_processed)
            result += (batches_concatenated,)

        if output_not_tuple:
            assert len(result) == 1
            result = result[0]

        return result
