import logging
import sys

from utils.prepare_run import prepare_run

if __name__ == '__main__':
    prepare_run()

    from configs import Config, Components
    from builder import Builder

    builder = Builder()

    if not Config.test_mode:
        builder.init_log_folder()
        builder.add_file_log_handler()

    builder.add_stream_log_handler()
    builder.create_logger()

    command = 'python ' + ' '.join(sys.argv)
    logging.info(f'Executing: {command}')

    if not Config.test_mode:
        builder.create_checkpointer()

    if Config.log_tensorboard and not Config.test_mode:
        builder.create_tensorboard_writer()
    if Config.log_terminal or Config.test_mode:
        builder.create_terminal_writer()
    builder.create_reporter()

    if Config.restore:
        builder.find_restoration_path()

    builder.create_dataset()
    builder.create_models()
    builder.create_optimizers()

    if Config.opt_level is not None:
        builder.apply_mixed_precision()

    if Config.restore:
        builder.restore_optimizers()
        if Config.opt_level is not None:
            builder.restore_amp()

    train = Components.train(builder.dataset, builder.models, builder.optimizers,
                             builder.reporter, builder.checkpointer)

    try:
        train.choochoo()
    except BaseException as ex:
        logging.exception(f'Exception: {ex} \n raised during training')
        if not Config.test_mode:
            # from configs import Status
            # builder.checkpointer.save(builder.models, builder.optimizers, Status.time)
            pass
    finally:
        if builder.tb_writer is not None:
            builder.tb_writer.tb_writer.close()
