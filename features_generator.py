import os
import logging.config
from logging.config import fileConfig
import pandas as pd
import numpy as np
import datetime as dt

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
today = dt.date.today()


class Features:

    def __init__(self, df):
        """
        :param df: pd.DataFrame()
        """
        if df is None or len(df) == 0:
            raise ValueError('You passed an empty dataframe.')

        self.df = df
        self.df_team = pd.DataFrame()

    def jornada_generator(self):
        """
        Take jornada from dirty data.

        :return:
        """

        df = self.df.copy()

        if 'jornada' not in df.columns:
            logger.warning('Cannot find "jornada" column in dataframe')
            return

        # non-critical field
        try:
            df['jornada'] = df['jornada'].apply(lambda x: int(x.split('regular_a_')[1].split('/')[0]))
        except ValueError:
            logger.warning('Cannot extract jornada number from url string')
            return

        if df is None or len(df) == 0:
            logger.warning('Dataframe is empty')
            return
        else:
            self.df = df.copy()

        return

    def local_visitante(self):
        """
        Take local and visitor team

        :return:
        """
        df = self.df.copy()

        if 'match' not in df.columns:
            raise ValueError('Cannot find "match" column in dataframe')

        try:
            df['local'] = df['match'].apply(lambda x: x.split('-')[0].strip())
            df['visitante'] = df['match'].apply(lambda x: x.split('-')[1].split('en directo')[0].strip())
        except ValueError as err:
            raise err

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        self.df = df.drop('match', axis=1).copy()

        return

    def goles(self):
        """
        Split result in goals as local and visitant

        result
        3-1

        local_goals, visitante_goals
        3, 1

        :return:
        """

        df = self.df.copy()

        if 'result' not in df.columns:
            raise ValueError('Cannot find "result" column in dataframe')

        try:
            df['local_goals'] = df['result'].apply(lambda x: int(x.split('-')[0].strip()))
            df['visitante_goals'] = df['result'].apply(lambda x: int(x.split('-')[1].strip()))
        except ValueError as err:
            raise err

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        self.df = df.drop('result', axis=1).copy()

        return

    def ganador(self):
        """
        Winner in quiniela format

        1 = Local is the winner
        X = Tie
        2 = Vistor is the winner

        :return:
        """

        df = self.df.copy()

        if 'local_goals' not in df.columns or 'visitante_goals' not in df.columns:
            raise ValueError('Cannot find "goals" columns in dataframe')

        try:
            df['winner'] = np.where(df['local_goals'] > df['visitante_goals'],
                                    1, np.where(df['local_goals'] == df['visitante_goals'],
                                                0,
                                                2))
        except ValueError as err:
            raise err

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        self.df = df.copy()

        return

    def team_by_team(self):
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

        :return:
        """

        df = self.df.copy()
        df_team = pd.DataFrame()

        cols_to_keep = ['team', 'jornada', 'GF', 'GC', 'score', 'local', 'winner']
        cols_init = ['local', 'jornada', 'local_goals', 'visitante_goals']

        if len([1 for col in cols_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        if 'winner' not in df.columns:
            try:
                self.ganador()
                df = self.df.copy()
            except ValueError:
                raise ValueError('"winner" column is not in dataframe')

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

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        self.df_team = df_team.copy()

        return

    @staticmethod
    def clasificacion(df, df_prev_jornada):
        """
        Create all classification stats, namely:

        https://resultados.as.com/resultados/futbol/primera/clasificacion/

        :param df:
        :param df_prev_jornada:
        :return:
        """

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        if df_prev_jornada is None:
            logger.warning('Cannot compute clasification with empty df_prev_jornada... Creating an empty dataframe')
            df_prev_jornada = pd.DataFrame()
            pass

        cols_init = ['winner', 'local', 'GF', 'GC', 'score']
        if len([1 for col in cols_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

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

        if df_last is None or len(df_last) == 0:
            raise ValueError('Cannot create dataframe of teams')

        if len(df_prev_jornada) != 0:
            if 'ranking' in df_prev_jornada.columns:
                df_prev_jornada.drop('ranking', axis=1, inplace=True)

            df_last = pd.concat([df_prev_jornada, df_last])
            df_last = df_last.groupby('team', as_index=False).sum()
            df_last['jornada'] = df['jornada'].unique()[0]

        df_last = df_last.sort_values(by='pts', ascending=False).reset_index(drop=True)
        df_last = df_last.reset_index().rename(columns={'index': 'ranking'})
        df_last['ranking'] = df_last['ranking']+1

        return df_last

# TODO. Falta crear el df_result
    @staticmethod
    def predictor_dataset(df, df_result):
        """
        Create dataframe of features for the predictor

        First create the ratio values of calendar dataframe
        Second merge both dataframes one with the output the second one with the features.

        :param df:
        :param df_result:
        :return:
        """

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        if df_result is None:
            raise ValueError('Dataframe is empty')

        cols_init = ['PG', 'PJ', 'PE', 'PP',
                     'PG_local', 'PJ_local', 'PE_local', 'PP_local',
                     'PG_visitante', 'PJ_visitante', 'PE_visitante', 'PP_visitante']
        cols_result_init = ['team', 'jornada']

        if len([1 for col in cols_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        if len([1 for col in cols_result_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        if len([1 for col in cols_result_init if col not in df_result.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        # global stats
        df['PG'] = df['PG']/df['PJ']
        df['PE'] = df['PE']/df['PJ']
        df['PP'] = df['PP']/df['PJ']
        # goals ratio
        df['GF_ratio'] = df['GF']/df['PJ']
        df['GC_ratio'] = df['GC']/df['PJ']

        # local stats
        df['PG_local'] = df['PG_local']/df['PJ_local']
        df['PE_local'] = df['PE_local']/df['PJ_local']
        df['PP_local'] = df['PP_local']/df['PJ_local']
        df['GF_ratio_local'] = df['GF_local']/df['PJ_local']
        df['GC_ratio_local'] = df['GC_local']/df['PJ_local']

        # visitante stats
        df['PG_visitante'] = df['PG_visitante']/df['PJ_visitante']
        df['PE_visitante'] = df['PE_visitante']/df['PJ_visitante']
        df['PP_visitante'] = df['PP_visitante']/df['PJ_visitante']
        df['GF_ratio_visitante'] = df['GF_ratio_visitante']/df['PJ_visitante']
        df['GC_ratio_visitante'] = df['GC_ratio_visitante']/df['PJ_visitante']

        df.drop(cols_init, axis=1, inplace=True)

# TODO. Acabar esto mergear
        df = df.merge(df_results, right_on=['jornada', 'team'], left_on=['jornada', 'team'])

        return df


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

    file_path = '{k}{f}'.format(k=key,
                                f=filename)

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
        logger.info('File {} created'.format(file_path))
    else:
        logger.warning('File {} already exists'.format(file_path))

    return


def main():

    # read matches
    df = pd.read_csv(input_file)

    if df is None or len(df) == 0:
        logger.error('Datraframe is null or empty')
        return

    featex = Features(df)

    # create stats
    featex.jornada_generator()
    featex.local_visitante()
    featex.goles()
    featex.ganador()

    key = './files/partit_a_partit/{y}/{m}/{d}/'.format(y=today.year,
                                                        m=today.month,
                                                        d=today.day)
    filename = 'partit_a_partit.csv'
    load_files(key, filename, featex.df)

    featex.team_by_team()

    df_prev_jornada = pd.DataFrame()

    for jornada in sorted(featex.df_team.jornada.unique()):

        df_temp = featex.clasificacion(featex.df_team[featex.df_team['jornada'] == jornada],
                                       df_prev_jornada)

        if df_temp is not None:
            df_prev_jornada = df_temp.copy()
        else:
            logger.error('clasificacion returned None')
            break

        key = './files/clasificacion/{y}/{m}/{d}/'.format(y=today.year,
                                                          m=today.month,
                                                          d=today.day)
        filename = 'clasificacion_{j}.csv'.format(j=jornada)
        load_files(key, filename, df_temp)

# TODO falta definir el df_result!! WiP
        predictor_dataset(df_temp, df_result)

        key = './files/predictor_dataset/{y}/{m}/{d}/'.format(y=today.year,
                                                              m=today.month,
                                                              d=today.day)
        filename = 'predictor_dataset_{j}.csv'.format(j=jornada)
        load_files(key, filename, df_temp)


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
