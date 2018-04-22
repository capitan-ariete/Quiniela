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

# TODO esto tiene que ser env variable
input_file = './files/file.csv'


def jornada_generator(df):
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
        df['jornada'] = df['jornada'].apply(lambda x: int(x.split('regular_a_')[1].split('/')[0]))
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


def team_by_team(df):
    """
    Split matches by team

    jornada, local, visitante, resultado, winner
    1, A, B, 3-0, 1

    Becomes

    jornada, team, goals, score
    1, A, 3, 3
    1, B, 0, 0

    where score is 3 if win 1 if tie 0 if loss

    team, score

    :param df:
    :return:
    """
    cols_to_keep = ['team', 'jornada', 'GF', 'GC', 'score', 'local', 'winner']

    df_team = pd.DataFrame()

    for local_team in df.local.unique():
        df_team_temp = df[df['local'] == local_team].copy()
        df_team_temp['score'] = np.where(df_team_temp['winner'] == 1,
                                         3,
                                         np.where(df_team_temp['winner'] == 0,
                                                  1,
                                                  0))

        df_team_temp['team'] = df_team_temp['local']
        df_team_temp['GF'] = df_team_temp['local_goals']
        df_team_temp['GC'] = df_team_temp['visitante_goals']
        df_team_temp['local'] = True
        df_team = pd.concat([df_team, df_team_temp[cols_to_keep]])

    for visitante_team in df.visitante.unique():
        df_team_temp = df[df['visitante'] == visitante_team].copy()
        df_team_temp['score'] = np.where(df_team_temp['winner'] == 2,
                                         3,
                                         np.where(df_team_temp['winner'] == 0,
                                                  1,
                                                  0))
        df_team_temp['team'] = df_team_temp['visitante']
        df_team_temp['GF'] = df_team_temp['visitante_goals']
        df_team_temp['GC'] = df_team_temp['local_goals']
        df_team_temp['goals'] = df_team_temp['visitante_goals']
        df_team_temp['local'] = False
        df_team = pd.concat([df_team, df_team_temp[cols_to_keep]])

    return df_team


def clasificacion(df, df_prev_jornada):
    """
    Create all classification stats, namely:

    https://resultados.as.com/resultados/futbol/primera/clasificacion/

    :param df:
    :param df_prev_jornada:
    :return:
    """
    df_last = df.copy()

    df_last['PJ'] = 1

    # local stats
    df_last['PG_local'] = np.where(df_last['winner'] == 1, np.where(df_last['local'], 1, 0), 0)
    df_last['PE_local'] = np.where(df_last['winner'] == 0, np.where(df_last['local'], 1, 0), 0)
    df_last['PP_local'] = np.where(df_last['winner'] == 2, np.where(df_last['local'], 1, 0), 0)
    df_last['GF_local'] = np.where(df_last['local'], df_last['GF'], 0)
    df_last['GC_local'] = np.where(df_last['local'], df_last['GC'], 0)
    df_last['pts_local'] = np.where(df_last['local'],
                                    df_last['score'],
                                    0)

    # visitante stats
    df_last['PG_visitante'] = np.where(df_last['winner'] == 2, np.where(~df_last['local'], 1, 0), 0)
    df_last['PE_visitante'] = np.where(df_last['winner'] == 0, np.where(~df_last['local'], 1, 0), 0)
    df_last['PP_visitante'] = np.where(df_last['winner'] == 1, np.where(~df_last['local'], 1, 0), 0)
    df_last['GF_visitante'] = np.where(~df_last['local'], df_last['GF'], 0)
    df_last['GC_visitante'] = np.where(~df_last['local'], df_last['GC'], 0)
    df_last['pts_visitante'] = np.where(~df_last['local'],
                                        df_last['score'],
                                        0)

    # global stats
    df_last['PG'] = df_last['PG_local'] + df_last['PG_visitante']
    df_last['PE'] = df_last['PE_local'] + df_last['PE_visitante']
    df_last['PP'] = df_last['PP_local'] + df_last['PP_visitante']
    df_last['pts'] = df_last['pts_local'] + df_last['pts_visitante']

    df_last.drop(['score', 'local', 'winner'], axis=1, inplace=True)

    if len(df_prev_jornada) != 0:
        df_last = pd.concat([df_prev_jornada, df_last])
        df_last = df_last.groupby('team', as_index=False).sum()
        df_last['jornada'] = df['jornada'].unique()[0]

    df_last = df_last.sort_values(by='pts', ascending=False)

    return df_last


def main():

    # read matches
    df = pd.read_csv(input_file)

    if df is None or len(df) == 0:
        logger.error('Datraframe is null or empty')
        return

    df = jornada_generator(df)

    if df is None or len(df) == 0:
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

    filename = './files/partit_a_partit.csv'
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
        logger.info('File {} created'.format(filename))
    else:
        logger.warning('File {} already exists'.format(filename))

    df_team = team_by_team(df)

    if df_team is None or len(df_team) == 0:
        logger.error('Cannot create dataframe of teams')
        return

    df_prev_jornada = pd.DataFrame()

    for jornada in sorted(df_team.jornada.unique()):
        df_temp = clasificacion(df_team[df_team['jornada'] == jornada], df_prev_jornada)
        df_prev_jornada = df_temp.copy()

        filename = './files/clasificacion_{}.csv'.format(jornada)
        if not os.path.isfile(filename):
            df_temp.to_csv(filename, index=False)
            logger.info('File {} created'.format(filename))
        else:
            logger.warning('File {} already exists'.format(filename))


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
