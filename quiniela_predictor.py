import os
import logging.config
from logging.config import fileConfig
import pandas as pd
from tpot import TPOTClassifier

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

train_file_seed = './files/predictor_dataset/2018/4/26/predictor_dataset_'
test_file_seed = './files/predictor_dataset_result/2018/4/26/predictor_dataset_result'


# TODO
def clean_before_tpot(X_train, y_train, X_test, y_test):
    return X_train, y_train, X_test, y_test


def main():

    X_train, y_train, X_test, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for j in range(8, 31):
        try:
            X_train = pd.concat([X_train,
                                 pd.read_csv('{f}{j}.csv'.format(f=train_file_seed, j=j))])
            y_train = pd.concat([y_train,
                                 pd.read_csv('{f}{j}.csv'.format(f=test_file_seed, j=j))])
        except ValueError:
            pass

    for j in range(31, 39):
        try:
            X_test = pd.concat([X_test,
                                pd.read_csv('{f}{j}.csv'.format(f=train_file_seed, j=j))])
            y_test = pd.concat([y_test,
                                pd.read_csv('{f}result{j}.csv'.format(f=train_file_seed, j=j))])
        except ValueError:
            pass

    X_train, y_train, X_test, y_test = clean_before_tpot(X_train, y_train, X_test, y_test)

    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_quiniela_pipeline.py')

    return


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
