import logging
import sys

from utils.prepare_run import prepare_run

if __name__ == '__main__':
    prepare_run()

    from configs import Config, Components
    from builder import Builder

    builder = Builder()

    builder.add_stream_log_handler()
    builder.create_logger()

    command = 'python ' + ' '.join(sys.argv)
    logging.info(f'Executing: {command}')

    if Config.restore:
        builder.find_restoration_path()

    builder.create_models()

    train = Components.interact(builder.models)

    try:
        train.choochoo()
    except BaseException as ex:
        logging.exception(f'Exception: {ex} \n raised during training')
