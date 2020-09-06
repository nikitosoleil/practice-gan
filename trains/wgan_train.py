import logging
from typing import Dict, Iterable, Iterator

import torch
from bert_score import BERTScorer
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from configs import Config, Status
from iterators import PerpetualLoader
from models import BaseModel
from utils.checkpointer import Checkpointer
from utils.misc import train_test_split
from utils.pone import Pone
from utils.reporter import Reporter
from utils.scheduler import Scheduler
from .locomotive import Locomotive


# emulate deterministic start tokens via high logits values
def emulate_logits(fake_logits, input_ids_t):
    batch_size, input_length = input_ids_t.shape
    i1 = torch.arange(batch_size).unsqueeze(1).repeat((1, input_length))
    i2 = torch.arange(input_length).unsqueeze(0).repeat((batch_size, 1))
    fake_logits[i1, i2, input_ids_t] = 1e3


class WGANTrain(Locomotive):
    def __init__(self, dataset: Dataset, models: Dict[str, BaseModel], optimizers: Dict[str, Optimizer],
                 reporter: Reporter, checkpointer: Checkpointer):
        super().__init__(dataset, models, optimizers, reporter, checkpointer)

        self.train_data, self.val_data = train_test_split(self.dataset)

        def collate_lm(list_of_pairs):
            input_ids, token_types = zip(*[self.models['generator'].prepare_input(s, finish=True)
                                           for s in list_of_pairs])
            aligned = self.models['generator'].align_right(input_ids, token_types)
            input_ids_t, token_types_t, attention_mask_t = map(torch.cat, aligned)
            return (), {'input_ids': input_ids_t, 'labels': input_ids_t,
                        'token_type_ids': token_types_t, 'attention_mask': attention_mask_t}

        def collate_validation(list_of_pairs):
            input_ids, token_types = zip(*[self.models['generator'].prepare_input(s[:1], finish=False)
                                           for s in list_of_pairs])
            aligned = self.models['generator'].align_right(input_ids, token_types)
            input_ids_t, token_types_t, attention_mask_t = map(torch.cat, aligned)
            return (), {'input_ids': input_ids_t, 'labels': input_ids_t,
                        'token_type_ids': token_types_t, 'attention_mask': attention_mask_t,
                        'real': [s[1] for s in list_of_pairs]}

        def collate_generation(list_of_pairs):
            start_input_ids, start_token_types = zip(*[self.models['generator'].prepare_input(s[:1], finish=False)
                                                       for s in list_of_pairs])
            aligned = self.models['generator'].align_right(start_input_ids, start_token_types)
            start_input_ids_t, start_token_types_t, start_attention_mask_t = map(torch.cat, aligned)
            return start_input_ids_t, start_token_types_t, start_attention_mask_t

        def collate_discrimination(list_of_pairs):
            start_input_ids_t, start_token_types_t, start_attention_mask_t = collate_generation(list_of_pairs)
            fake_input_ids, fake_logits = self.models['generator'].generate(start_input_ids=start_input_ids_t,
                                                                            start_token_type_ids=start_token_types_t,
                                                                            start_attention_mask=start_attention_mask_t,
                                                                            with_logits=True, with_grad=False)
            emulate_logits(fake_logits, start_input_ids_t)
            true_input_ids, true_token_types = zip(*[self.models['discriminator'].prepare_input(s, finish=True)
                                                     for s in list_of_pairs])
            true_input_ids_t = torch.cat(true_input_ids)
            all_input_ids = torch.cat([true_input_ids_t, fake_input_ids])
            attention_mask = (all_input_ids != self.models['generator'].pad).float()
            return (), {'true_input_ids': true_input_ids_t, 'fake_logits': fake_logits, 'attention_mask': attention_mask}

        collate_fns = {'lm': collate_lm, 'discrimination': collate_discrimination, 'generation': collate_generation}

        self.train_loader: Dict[str, DataLoader] = dict()
        self.val_loader: Dict[str, DataLoader] = dict()

        for phase in ['lm', 'discrimination']:
            self.train_loader[phase] = DataLoader(self.train_data, batch_size=Config.batch_size[phase],
                                                  collate_fn=collate_fns[phase], shuffle=True)  # , num_workers=Config.n_cores)
            self.val_loader[phase] = DataLoader(self.val_data, batch_size=Config.batch_size[phase],
                                                collate_fn=collate_fns[phase], shuffle=True)  # , num_workers=Config.n_cores)

        self.train_source: Dict[str, Iterator] = dict()
        self.val_source: Dict[str, Iterable] = dict()

        self.train_source['lm'] = PerpetualLoader(self.train_loader['lm'])
        self.val_source['lm'] = self.val_loader['lm']

        self.train_source['discrimination'] = PerpetualLoader(self.train_loader['discrimination'])
        self.val_source['discrimination'] = self.val_loader['discrimination']

        self.train_source['generation'] = PerpetualLoader(DataLoader(self.train_data, batch_size=Config.batch_size['generation'],
                                                                     collate_fn=collate_generation, shuffle=True))

        self.val_source['validation'] = DataLoader(self.val_data, batch_size=Config.val_samples,
                                                   collate_fn=collate_validation, shuffle=True)

        if Config.pone_validation:
            self.pone = Pone(models)
        if Config.bert_score_validation:
            self.bert_score = BERTScorer(nthreads=Config.n_cores, lang='en', device='cuda:0', rescale_with_baseline=True)

    def choochoo(self):
        logging.info('Training started')

        with trange(Status.time + 1, Status.time + Config.training_steps + 1, desc='Steps') as t:
            for Status.time in t:
                # if Status.time == 395:
                #     print(Status.cache['length'])
                #     import pickle
                #     with open('length.pkl', 'wb') as file:
                #         pickle.dump(Status.cache['length'], file)

                for phase in ['lm', 'discrimination']:
                    model = Config.phase_models[phase]

                    if Config.is_running[phase]:
                        t.set_postfix_str(phase)

                        for i in range(Config.batches_per_step[phase]):
                            args, kwargs = next(self.train_source[phase])
                            # print(args, kwargs)
                            loss, stats = self.models[model].train(*args, **kwargs)

                            self.loose(loss, model, phase, i)
                            self.reporter.full_report('training', phase, loss, stats)

                # GENERATOR

                phase = 'generation'
                model = Config.phase_models[phase]

                if Config.is_running[phase]:
                    t.set_postfix_str(phase)

                    for i in range(Config.batches_per_step[phase]):
                        start_input_ids_t, start_token_types_t, start_attention_mask_t = next(self.train_source['generation'])

                        self.models[model].network.train()
                        fake_input_ids, fake_logits = self.models[model].generate(start_input_ids=start_input_ids_t,
                                                                                  start_token_type_ids=start_token_types_t,
                                                                                  start_attention_mask=start_attention_mask_t,
                                                                                  with_logits=True, with_grad=True)
                        emulate_logits(fake_logits, start_input_ids_t)
                        attention_mask = (fake_input_ids != self.models[model].pad).float()
                        discriminator_logits = self.models['discriminator'].forward(fake_logits=fake_logits,
                                                                                    attention_mask=attention_mask)[0]
                        loss = -discriminator_logits.mean()

                        self.loose(loss, model, phase, i)
                        self.optimizers['discriminator'].zero_grad()
                        self.reporter.full_report('training', phase, loss, {'score_mean': discriminator_logits.mean()})

                if Scheduler.is_logging():
                    self.reporter.push('training')

                # VALIDATION

                for phase in ['lm', 'discrimination']:
                    model = Config.phase_models[phase]

                    if Scheduler.is_validating(model):
                        t.set_postfix_str(f'validating {model}')

                        for args, kwargs in self.val_source[phase]:
                            loss, stats = self.models[model].validate(*args, **kwargs)
                            self.reporter.full_report('validation', phase, loss, stats)

                if Scheduler.is_validating():
                    t.set_postfix_str(f'validating all')

                    s = ''
                    queries, responses, real = [], [], []

                    for batch, (args, kwargs) in enumerate(self.val_source['validation']):
                        if batch == Config.val_all_batches:
                            break

                        start_input_ids = kwargs['input_ids']
                        start_token_type_ids = kwargs['token_type_ids']
                        start_attention_mask = kwargs['attention_mask']
                        real += kwargs['real']

                        input_ids, generator_logits = self.models['generator'].generate(start_input_ids=start_input_ids,
                                                                                        start_token_type_ids=start_token_type_ids,
                                                                                        start_attention_mask=start_attention_mask,
                                                                                        with_logits=True, with_grad=False)
                        attention_mask = (input_ids != self.models['generator'].pad)
                        discriminator_logits = self.models['discriminator'].forward(fake_logits=generator_logits,
                                                                                    attention_mask=attention_mask.float())[0]

                        batch_size, input_length = start_input_ids.shape
                        input_ids = input_ids[:, input_length:]
                        attention_mask = attention_mask[:, input_length:]

                        for i in range(Config.val_samples):
                            query = start_input_ids[i][start_attention_mask[i].bool()]
                            response = input_ids[i][attention_mask[i]]

                            query = query[2:-1]  # <bos> <speaker2> Text <speaker1>
                            response = response[:-1]  # Text <eos>

                            query_s, response_s = map(self.models['generator'].convert_output, (query, response))
                            queries.append(query_s)
                            responses.append(response_s)

                            if batch == 0:
                                s += f'{query_s} | {response_s} | {discriminator_logits[i].item()} \n\n'

                    if Config.pone_validation:
                        pone_score = self.pone.score(queries, responses)
                        self.reporter.report('validation', 'lm', 'pone', pone_score)

                    if Config.bert_score_validation:
                        P, R, bert_score = self.bert_score.score(responses, real, verbose=False)
                        self.reporter.report('validation', 'lm', 'bert_score', bert_score.mean().item())
                        # print(bert_score)

                    self.reporter.direct_report('sampling', s)

                    self.reporter.push('validation')

                if Scheduler.is_checkpointing():
                    self.checkpointer.save(self.models, self.optimizers, Status.time)
