import logging
import datetime
import os


def setup_logger(args):

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

    os.makedirs(args.logdir, exist_ok=True)  # Create the directory if it doesn't exist
    file_handler = logging.FileHandler(os.path.join(args.logdir, f'logFile_{current_datetime}.log'))

    # file_handler = logging.FileHandler(f'logFile_{current_datetime}.log')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
