import datetime as dt
import logging.config
import pandas as pd
from tpot import TPOTClassifier
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import ExtraTreesClassifier
from features import Features

logger = logging.getLogger(__name__)

today = dt.date.today()
today_folder = '{y}/{m}/{d}/'.format(y=today.year,
                                     m=today.month,
                                     d=today.day)

x_file_seed = '../files/segunda/predictor_dataset/{today}'.format(today=today_folder)
y_file_seed = '../files/segunda/predictor_dataset_result/{today}'.format(today=today_folder)

x_file_seed = [(f, join(x_file_seed, f))
               for f in listdir(x_file_seed) if isfile(join(x_file_seed, f))]
y_file_seed = [(f, join(y_file_seed, f))
               for f in listdir(y_file_seed) if isfile(join(y_file_seed, f))]

# TODO hay que encontrar una manera de no hardcodear este numero y que lo coja segun la liga
jornada_limite = 36


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

    X_train, X_test = read_initial_datasets(x_file_seed)
    y_train, y_test = read_initial_datasets(y_file_seed)

    # initialize the class with a dummy dataframe
    featex = Features(X_train)
    X_train, y_train, X_test, y_test = featex.clean_before_prediction(X_train,
                                                                      y_train,
                                                                      X_test,
                                                                      y_test)

    # tpot_generation(X_train, y_train, X_test, y_test)

# TODO. Esto es sacado de lo que genera tpot. Hay que encontrar una manera de lanzarlo mas elegante
    exported_pipeline = ExtraTreesClassifier(bootstrap=True,
                                             criterion="entropy",
                                             max_features=0.9000000000000001,
                                             min_samples_leaf=12,
                                             min_samples_split=10,
                                             n_estimators=100)

    exported_pipeline.fit(X_train, y_train)
# TODO hay que encontrar una manera de guardar este result
    results = exported_pipeline.predict(X_test)

    return
