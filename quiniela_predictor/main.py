import os

import logging.config
from logging.config import fileConfig

from features_generator import features_generator_orchestrator
from quiniela_predictor import quiniela_predictor_orchestrator

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():
    """Main function
    """
    try:
        features_generator_orchestrator()
        logger.info('Features created.')
    except ValueError:
        raise ValueError('Cannot finish feature generation.')

    try:
        quiniela_predictor_orchestrator()
        logger.info('Prediction done.')
    except ValueError:
        raise ValueError('Cannot finish prediction.')


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
