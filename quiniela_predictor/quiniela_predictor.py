import datetime as dt
import logging.config
import pandas as pd
from tpot import TPOTClassifier
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import ExtraTreesClassifier
from features import Features

logger = logging.getLogger(__name__)

ligas = ['primera', 'segunda']
prediction = ['match_', '']

today = dt.date.today()
today_folder = '{y}/{m}/{d}/'.format(y=today.year,
                                     m=today.month,
                                     d=today.day)

jornada_inicial = 8
jornada_limite = 34


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
        elif jornada >= jornada_limite:
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

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2, n_jobs=4)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_quiniela_pipeline.py')

    return


def main():

    for liga in ligas:
        for p in prediction:

            file_seed = f'./files/{liga}/predictor_{p}dataset/{today_folder}'

            file_seed = [(f, join(file_seed, f))
                         for f in listdir(file_seed) if isfile(join(file_seed, f))]

            train, test = read_initial_datasets(file_seed)

            X_train = train.drop(['winner', 'jornada'], axis=1)
            #X_test = test.drop(['winner', 'jornada'], axis=1)
            X_test = test.drop(['jornada'], axis=1)
            y_train = train[['winner']]
            #y_test = test[['winner']]

            # initialize the class with a dummy dataframe
            featex = Features(X_train)
            X_train, X_test = featex.clean_before_prediction(X_train,
                                                             X_test)

            #tpot_generation(X_train, y_train, X_test, y_test)

# TODO. Esto es sacado de lo que genera tpot. Hay que encontrar una manera de lanzarlo mas elegante
            exported_pipeline = ExtraTreesClassifier(bootstrap=True,
                                                     criterion="entropy",
                                                     max_features=0.9,
                                                     min_samples_leaf=12,
                                                     min_samples_split=10,
                                                     n_estimators=100)

            exported_pipeline.fit(X_train, y_train)
# TODO hay que guardar este result
            results = exported_pipeline.predict(X_test)

            print(list(zip(results, y_test)))

    return
