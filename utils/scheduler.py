from configs import Config, Status


class Scheduler:
    @staticmethod
    def __is_training(model):
        return any(Config.is_running[phase] for phase in Config.model_phases[model])

    @staticmethod
    def is_validating(model=None):
        return (Status.time + 1) % Config.val_freq == 0 and \
               (Scheduler.__is_training(model) if model is not None else True)

    @staticmethod
    def is_logging(phase=None):
        return (Status.time + 1) % Config.train_log_freq == 0 and \
               (Config.is_running[phase] if phase is not None else True)

    @staticmethod
    def is_checkpointing(model=None):
        return (Status.time + 1) % Config.checkpoint_freq == 0 and \
               (Scheduler.__is_training(model) if model is not None else True)
