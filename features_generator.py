import os
import logging.config
from logging.config import fileConfig
import pandas as pd
import numpy as np

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

input_file = './files/file.csv'


def jornada(df):
    """
    Take jornada from dirty data.

    :param df:
    :return:
    """
    if 'jornada' not in df.columns:
        logger.warning('Cannot find "jornada" column in dataframe')
        return

    # non-critical field
    try:
        df['jornada'] = df['jornada'].apply(lambda x: x.split('regular_a_')[1].split('/')[0])
    except ValueError:
        logger.warning('Cannot extract jornada number from url string')

    return df


def local_visitante(df):
    """
    Take local and visitor team

    :param df:
    :return:
    """

    if 'match' not in df.columns:
        logger.warning('Cannot find "match" column in dataframe')
        return

    try:
        df['local'] = df['match'].apply(lambda x: x.split('-')[0].strip())
        df['visitante'] = df['match'].apply(lambda x: x.split('-')[1].split('en directo')[0].strip())
    except ValueError:
        logger.warning('Cannot extract jornada number from url string')
        return

    return df.drop('match', axis=1)


def goles(df):
    """

    :param df:
    :return:
    """

    if 'result' not in df.columns:
        logger.warning('Cannot find "result" column in dataframe')
        return

    try:
        df['local_goals'] = df['result'].apply(lambda x: int(x.split('-')[0].strip()))
        df['visitante_goals'] = df['result'].apply(lambda x: int(x.split('-')[1].strip()))
    except ValueError:
        logger.warning('Cannot extract goles')
        return

    return df.drop('result', axis=1)


def ganador(df):
    """
    Winner in quiniela format

    1 = Local is the winner
    X = Tie
    2 = Vistor is the winner

    :param df:
    :return:
    """

    if 'local_goals' not in df.columns or 'visitante_goals' not in df.columns:
        logger.warning('Cannot find "goals" columns in dataframe')
        return

    try:
        df['winner'] = np.where(df['local_goals'] > df['visitante_goals'],
                                1, np.where(df['local_goals'] == df['visitante_goals'],
                                            0,
                                            2))
    except ValueError:
        logger.warning('Cannot find winner')
        return

    return df


# TODO. To develop.
def clasificacion(df, df_prev_jornada):
    """
    Create clasification.

    :param df:
    :param df_prev_jornada:
    :return:
    """

    return df


def main():

    # read matches
    df = pd.read_csv(input_file)

    if df is None or len(df)==0:
        logger.error('Datraframe is null or empty')
        return

    df = jornada(df)

    if df is None or len(df)==0:
        logger.error('Cannot generate jornadas')
        return

    df = local_visitante(df)

    if df is None or len(df) == 0:
        logger.error('Cannot generate local and visitor teams')
        return

    df = goles(df)

    if df is None or len(df) == 0:
        logger.error('Cannot generate local and visitor teams')
        return

    df = ganador(df)

    if df is None or len(df) == 0:
        logger.error('Cannot generate local and visitor teams')
        return

    df_prev_jornada = pd.DataFrame()

    for jornada in df.jornada.unique():
        df_temp = clasificacion(df[df['jornada'] == jornada], df_prev_jornada)

# TODO. Aqui falta guardar este df.

        df_prev_jornada = df_temp.copy()


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
