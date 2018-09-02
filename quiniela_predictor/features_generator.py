import os

import logging.config
import pandas as pd
import datetime as dt

from features import Features

logger = logging.getLogger(__name__)

input_file = './files/file.csv'
ligas = ['primera', 'segunda']
today = dt.date.today()


def load_files(key, filename, df):
    """
    Load files to folder.

    :param key: string. folder key
    :param filename: string. filename
    :param df: pd.DataFrame().
    :return:
    """
    if not os.path.isdir(key):
        os.makedirs(key)

    file_path = f'{key}{filename}'

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
        logger.info(f'File {file_path} created')
    else:
        logger.warning(f'File {file_path} already exists')


def _build_classification():
    """
    Build clasificacion dataset
    :return:
    """
    df_jornada = featex.df_team[featex.df_team['jornada'] == jornada]
    df_temp = featex.clasificacion(df_jornada, df_prev_jornada)

    if df_temp is not None:
        df_prev_jornada = df_temp.copy()
    else:
        logger.error('clasificacion returned None')
        break

    key = f'./files/{liga}/clasificacion/{today_folder}'
    filename = f'clasificacion_{jornada}.csv'
    load_files(key, filename, df_temp)


def _build_match_prediction():
    """
    Build match prediction dataset
    """
    df_predictor_match_dataset = featex.predictor_dataset_match(df_temp)
    key = f'./files/{liga}/predictor_match_dataset/{today_folder}'
    filename = f'predictor_match_dataset_{jornada}.csv'
    load_files(key, filename, df_predictor_match_dataset)


def _build_team_prediction():
    """
    Build team prediction dataset
    """
    df_predictor_dataset = featex.predictor_dataset_team_by_team(df_temp)

    key = f'./files/{liga}/predictor_dataset/{today_folder}'
    filename = f'predictor_dataset_{jornada}.csv'
    load_files(key, filename, df_predictor_dataset)


# TODO esto tiene que estar incluido en el predictor_data_set_{j}!! No separado!!
def _build_team_result():
    '''
    Build team result prediction dataset
    '''
    if len(df_predictor_dataset_old) > 0 and '1_match_ago' in df_predictor_dataset.columns:
        y = df_predictor_dataset_old.merge(df_predictor_dataset[['team', '1_match_ago']],
                                           how='left',
                                           on='team').\
            rename(columns={'1_match_ago': 'result'})

        key = f'./files/{liga}/predictor_dataset_result/{today_folder}'
        filename = f'predictor_dataset_result_{jornada}.csv'
        load_files(key, filename, y)

    return df_predictor_dataset[['team', 'jornada']]


def process_jornada(jornada):
    """
    Process every single jornada prediction
    :return:
    """
    _build_classification()
    _build_match_prediction()
    _build_team_prediction()
    df_predictor_dataset_old = _build_team_result()

    return df_predictor_dataset_old


def _build_results():
    """
    Build results dataset
    """
    today_folder = f'{today.year}/{today.month}/{today.day}/'

    key = f'./files/{liga}/partit_a_partit/{today_folder}'
    filename = 'partit_a_partit.csv'
    load_files(key, filename, featex.df)

    featex.team_by_team()

    jornadas = featex.df_team.jornada.unique()
    for jornada in sorted(jornadas):
        process_jornada(jornada)


def process_league(liga):
    """
    Process every single match for a given league.
    :warning: Working for La Liga and 2da division
    :return:
    """
    df = df_all[df_all['liga'] == liga]
    df.drop('liga', axis=1, inplace=True)

    featex = Features(df)

    # create stats
    featex.jornada_generator()
    featex.local_visitante()
    featex.goles()
    featex.ganador()

    build_results()


def features_generator_orchestrator():

    today = dt.date.today()

    # read matches
    df_all = pd.read_csv(input_file)

    if df_all is None or len(df_all) == 0:
        raise ValueError('Input dataset is null or empty')

    for liga in ligas:
        logger.info(f'Generating data for competition: {liga}')
        process_league(liga)
        logger.info(f'League {liga} | prediction completed')
