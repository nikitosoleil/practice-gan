# noinspection PyUnresolvedReferences
from configs.status import Status
from configs.system_config import SystemConfig, dd
from utils.classproperty import classproperty


class Config(SystemConfig):
    models = ['generator', 'discriminator']
    trainable_models = models
    device = {'generator': 'cuda:0',
              'discriminator': 'cuda:0'}

    data_path = 'data/mnist'

    phases = ['generation', 'discrimination']
    phase_models = {'generation': 'generator',
                    'discrimination': 'discriminator'}
    model_phases = {'generator': ['generation'],
                    'discriminator': ['discrimination']}

    # MODEL

    latent_dim = 100
    img_size = 32
    channels = 1
    clip_value = 0.01

    # TRAINING

    max_batch_size = {'generator': 1024,
                      'discriminator': 1024}
    batch_size = {'discrimination': 64,
                  'generation': 64}
    accumulation_steps = {'discrimination': 1,
                          'generation': 1}
    batches_per_step = {'discrimination': 5,
                        'generation': 1}
    learning_rate = dd(5e-5)
    max_norm = dd(1.0)
    opt_level = None  # 'O1'

    train_portion = 0.98
    training_steps = 100000
    val_samples = 25
    val_all_batches = 16

    # SCHEDULING

    train_log_freq = 5
    val_freq = 200
    checkpoint_freq = 1000

    @classproperty
    def is_running(self):
        return dd(True)

    @classproperty
    def is_stepping(self):
        return dd(True)

    # CHECKPOINTS

    restore = False
    restore_from = '-1'
    restore_time = '-1'

    seed = 1092020


# TODO: more per-model options
# TODO: console logs

from torch.optim import Adam
from models import BaseModel, Discriminator
from networks import GeneratorNN, DiscriminatorNN
from trains import GANTrain, GANInteract


class Components:
    dataset = lambda: None
    models = {'generator': BaseModel,
              'discriminator': Discriminator}
    networks = {'generator': GeneratorNN,
                'discriminator': DiscriminatorNN}
    optimizers = dd(Adam)
    train = GANTrain
    interact = GANInteract
