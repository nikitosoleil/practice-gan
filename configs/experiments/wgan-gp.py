# noinspection PyUnresolvedReferences
from configs.status import Status
from configs.system_config import SystemConfig, dd
from utils.classproperty import classproperty


class Config(SystemConfig):
    @classproperty
    def regularization_weight(self):
        # discrimination_times = Status.time - Config.discrimination_start_delay
        return 10.  # 0.1 if discrimination_times < 10 else 1. if discrimination_times < 20 else 10.

    loss_scale = 1.
