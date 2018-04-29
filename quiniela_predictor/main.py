import os
import logging.config
from logging.config import fileConfig
import features_generator as featex
import quiniela_predictor as predictin

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():
    try:
        featex.main()
    except ValueError:
        raise ValueError('Cannot finish feature generation.')

    try:
        predictin.main()
    except ValueError:
        raise ValueError('Cannot finish prediction.')

    return


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
