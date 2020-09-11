# noinspection PyUnresolvedReferences
from configs.status import Status
from configs.system_config import SystemConfig, dd
from utils.classproperty import classproperty


class Config(SystemConfig):
    @classproperty
    def regularization_weight(self):
        return 10.
