import os
import datetime as dt
import logging.config
from logging.config import fileConfig
import pandas as pd
from tpot import TPOTClassifier
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import ExtraTreesClassifier

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

today = dt.date.today()
today_folder = '{y}/{m}/{d}/'.format(y=today.year,
                                     m=today.month,
                                     d=today.day)

x_file_seed = './files/segunda/predictor_dataset/{today}'.format(today=today_folder)
y_file_seed = './files/segunda/predictor_dataset_result/{today}'.format(today=today_folder)

x_file_seed = [(f, join(x_file_seed, f))
               for f in listdir(x_file_seed) if isfile(join(x_file_seed, f))]
y_file_seed = [(f, join(y_file_seed, f))
               for f in listdir(y_file_seed) if isfile(join(y_file_seed, f))]

jornada_limite = 36


def transform(x):
    if x == 'W':
        return 3
    elif x == 'T':
        return 1
    else:
        return 0


def clean_before_tpot(X_train, y_train, X_test, y_test):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    X_train = X_train.fillna(0)
    y_train = y_train.fillna(0)
    X_test = X_test.fillna(0)
    y_test = y_test.fillna(0)

    X_train.drop(['team', 'jornada'], axis=1, inplace=True)
    y_train.drop(['team', 'jornada'], axis=1, inplace=True)
    X_test.drop(['team', 'jornada'], axis=1, inplace=True)
    y_test.drop(['team', 'jornada'], axis=1, inplace=True)

    win_labels = ['1_match_ago',
                  '2_match_ago',
                  '3_match_ago',
                  '4_match_ago',
                  '5_match_ago',
                  '6_match_ago',
                  '7_match_ago']

    for label in win_labels:
        X_train[label] = X_train[label].apply(lambda x: transform(x))
        X_test[label] = X_test[label].apply(lambda x: transform(x))

    y_train['result'] = y_train['result'].apply(lambda x: transform(x))
    y_test['result'] = y_test['result'].apply(lambda x: transform(x))

    return X_train, y_train, X_test, y_test


def read_initial_datasets(files_list):
    """

    :param files_list:
    :return:
    """
    train = pd.DataFrame()
    test = pd.DataFrame()

    for filename, file in files_list:
        try:
            jornada = int(filename.split('.csv')[0].split('_')[-1])
        except ValueError:
            raise ValueError('The files do not contain the jornada')

        if jornada in range(8, jornada_limite):
            try:
                train = pd.concat([train, pd.read_csv(file)])
            except ValueError:
                pass
        elif jornada >= jornada_limite:
            try:
                test = pd.concat([test, pd.read_csv(file)])
            except ValueError:
                pass
        else:
            continue

    return train, test


def tpot_generation(X_train, y_train, X_test, y_test):
    tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2, n_jobs=4)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_quiniela_pipeline.py')


def main():

    X_train, X_test = read_initial_datasets(x_file_seed)
    y_train, y_test = read_initial_datasets(y_file_seed)

    X_train, y_train, X_test, y_test = clean_before_tpot(X_train, y_train, X_test, y_test)

    #tpot_generation(X_train, y_train, X_test, y_test)

    exported_pipeline = ExtraTreesClassifier(bootstrap=True,
                                             criterion="entropy",
                                             max_features=0.9000000000000001,
                                             min_samples_leaf=12,
                                             min_samples_split=10,
                                             n_estimators=100)

    exported_pipeline.fit(X_train, y_train)
    results = exported_pipeline.predict(X_test)

    return


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
