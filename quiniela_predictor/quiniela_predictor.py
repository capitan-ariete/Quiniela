from os import listdir
from os.path import isfile, join

import datetime as dt
import logging.config
import pandas as pd
from tpot import TPOTClassifier
from sklearn.ensemble import RandomForestClassifier

from features import Features

logger = logging.getLogger(__name__)

ligas = ['primera', 'segunda']
prediction = ['match_', '']

today = dt.date.today()
today_folder = f'{today.year}/{today.month}/{today.day}/'

jornada_inicial = 8
jornada_limite = 30
otras_jornadas = 34


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

        if jornada in range(jornada_inicial, jornada_limite):
            try:
                train = pd.concat([train, pd.read_csv(file)])
            except ValueError:
                pass
        elif jornada >= jornada_limite and jornada < otras_jornadas:
            try:
                test = pd.concat([test, pd.read_csv(file)])
            except ValueError:
                pass
        else:
            continue

    return train, test


def tpot_generation(X_train, y_train, X_test, y_test):
    """
    TPOT module generation

    https://github.com/EpistasisLab/tpot

    :param X_train: training features
    :param y_train: training results
    :param X_test: test features
    :param y_test: test results (prediction)
    :return:
    """
    tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2, n_jobs=4)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_quiniela_pipeline.py')


def process_prediction(liga, p):
    """

    :param liga:
    :param p:
    :return:
    """
    file_seed = f'./files/{liga}/predictor_{p}dataset/{today_folder}'

    file_seed = [(f, join(file_seed, f))
                 for f in listdir(file_seed) if isfile(join(file_seed, f))]

    train, test = read_initial_datasets(file_seed)

    X_train = train.drop(['winner', 'jornada'], axis=1)
    X_test = test.drop(['winner', 'jornada'], axis=1)
    y_train = train[['winner']]
    y_test = test[['winner']]

    # initialize the class with a dummy dataframe
    featex = Features(X_train)
    X_train, X_test = featex.clean_before_prediction(X_train,
                                                     X_test)

    # tpot_generation(X_train, y_train, X_test, y_test)

    clf = RandomForestClassifier(n_estimators=500, criterion='entropy')
    clf.fit(X_train, y_train)
    results_proba = clf.predict_proba(X_test)
    results = clf.predict(X_test)
    score = clf.score(X_test, y_test)

    logger.info(f'League {liga} | '
                f'Prediction {p} | '
                f'final result: {list(zip(results, y_test))}')


def quiniela_predictor_orchestrator():
    """
    Orchestrates Prediction league by league
    :return:
    """
    for liga in ligas:
        for p in prediction:
            process_prediction(liga, p)
