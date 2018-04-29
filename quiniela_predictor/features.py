import pandas as pd
import numpy as np


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

    def match_ranking_diff(self, df_clasificacion, jornada):
        """
        Difference in the ranking between rivals

        For instance if

        local, visitante, jornada
        Barcelona, R.Madrid, 11

        and

        team, ranking, jornada
        Barcelona, 1, 10
        R.Madrid, 3, 10

        Then

        local, jornada, rival_ranking_diff
        Barcelona, 11, -2
        R.Madrid, 11, +2

        :param df_clasificacion: pandas DataFrame. clasificacion dataframe
        :param jornada: integer. Jornada to parse
        :return:
        """

        if self.df is None or len(self.df) == 0:
            raise ValueError('Dataframe is empty')

        if 'jornada' not in self.df.columns:
            raise ValueError('Dataframe has to have "jornada" feature')

        if jornada > 1:
            df_results = self.df[self.df['jornada'] == jornada - 1]
        else:
            df_results = self.df[self.df['jornada'] == jornada]

        df_temp = pd.concat([df_results,
                             df_results.rename(columns={'local': 'visitante',
                                                        'visitante': 'local'})])
        df_temp.rename(columns={'visitante': 'rival'}, inplace=True)
        df_temp = df_clasificacion.merge(df_temp[['local', 'rival']],
                                         how='left',
                                         left_on='team',
                                         right_on='local')

        df_clasificacion = df_temp.merge(df_clasificacion[['team', 'ranking']],
                                         how='left',
                                         left_on='rival',
                                         right_on='team').copy()
        df_clasificacion['rival_ranking_diff'] = df_clasificacion['ranking_x'] - df_clasificacion['ranking_y']
        df_clasificacion.drop(['local', 'rival', 'team_y', 'ranking_y'], axis=1, inplace=True)
        df_clasificacion.rename(columns={'ranking_x': 'ranking', 'team_x': 'team'}, inplace=True)

        return df_clasificacion

    def predictor_dataset_team_by_team(self, df_temp):
        """
        Create dataframe of features for the predictor

        First create the ratio values of calendar dataframe
        Second merge both dataframes one with the output the second one with the features.

        The final dataframe predicts whether a team wins a match or not

        For instance match between 1st and 5th:

        ranking, GF, ..., result
        1,       5,  ...,   X

        :param df: pandas DataFrame. Calendar dataframe
        :return:
        """

        df = df_temp.copy()

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        cols_init = ['PG', 'PJ', 'PE', 'PP', 'PG_local', 'PJ_local', 'PE_local', 'PP_local',
                     'PG_visitante', 'PJ_visitante', 'PE_visitante', 'PP_visitante']
        cols_result_init = ['team', 'jornada']

        if len([1 for col in cols_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        if len([1 for col in cols_result_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        # global stats
        df['PG'] = df['PG'] / df['PJ']
        df['PE'] = df['PE'] / df['PJ']
        df['PP'] = df['PP'] / df['PJ']
        # goals ratio
        df['GF_ratio'] = df['GF'] / df['PJ']
        df['GC_ratio'] = df['GC'] / df['PJ']

        # local stats
        df['PG_local'] = df['PG_local'] / df['PJ_local']
        df['PE_local'] = df['PE_local'] / df['PJ_local']
        df['PP_local'] = df['PP_local'] / df['PJ_local']
        df['GF_ratio_local'] = df['GF_local'] / df['PJ_local']
        df['GC_ratio_local'] = df['GC_local'] / df['PJ_local']

        # visitante stats
        df['PG_visitante'] = df['PG_visitante'] / df['PJ_visitante']
        df['PE_visitante'] = df['PE_visitante'] / df['PJ_visitante']
        df['PP_visitante'] = df['PP_visitante'] / df['PJ_visitante']
        df['GF_ratio_visitante'] = df['GF_visitante'] / df['PJ_visitante']
        df['GC_ratio_visitante'] = df['GC_visitante'] / df['PJ_visitante']

        df.drop(['GF', 'GC', 'PJ', 'PJ_local', 'PJ_visitante'], axis=1, inplace=True)

        df_results = self.df[self.df['jornada'] == df.jornada.values[0]]
        df_results.loc[:, 'is_local'] = 1

        df = df.merge(df_results[['local', 'is_local']],
                      how='left',
                      left_on='team',
                      right_on='local').drop(['local'], axis=1)

        df = df.fillna(0)

        # results is the result of following jornada
        jornada = df.jornada.values[0] - 1

        if jornada in range(1, 39):
            df = self.match_ranking_diff(df, jornada)

        return df

    def predictor_dataset_match(self, df_temp):
        """
        Create dataframe of features for the predictor

        First create the ratio values of calendar dataframe
        Second merge both dataframes one with the output the second one with the features.

        The final dataframe gives a winner for every match and have both teams statistics joined

        For instance match between 1st and 5th:

        ranking_x, ranking_y, GF_x, GF_y, ..., result
        1,         5,         9,    7,  ...,   X

        :param df: pandas DataFrame. Calendar dataframe
        :return:
        """

        df = df_temp.copy()

        if df is None or len(df) == 0:
            raise ValueError('Dataframe is empty')

        cols_init = ['PG', 'PJ', 'PE', 'PP',
                     'PG_local', 'PJ_local', 'PE_local', 'PP_local',
                     'PG_visitante', 'PJ_visitante', 'PE_visitante', 'PP_visitante']
        cols_result_init = ['team', 'jornada']

        if len([1 for col in cols_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        if len([1 for col in cols_result_init if col not in df.columns]) > 0:
            raise ValueError('Miss some mandatory columns in the dataframe')

        df['GF_ratio'] = df['GF'] / df['PJ']
        df['GC_ratio'] = df['GC'] / df['PJ']

        # local stats
        df['PG_local'] = df['PG_local'] / df['PJ_local']
        df['PE_local'] = df['PE_local'] / df['PJ_local']
        df['PP_local'] = df['PP_local'] / df['PJ_local']

        # visitante stats
        df['PG_visitante'] = df['PG_visitante'] / df['PJ_visitante']
        df['PE_visitante'] = df['PE_visitante'] / df['PJ_visitante']
        df['PP_visitante'] = df['PP_visitante'] / df['PJ_visitante']

        df.drop(['PJ', 'PJ_local', 'PJ_visitante', 'PG', 'PE', 'PP',
                 'GF_visitante', 'GC_visitante', 'pts_visitante',
                 'GF_local', 'GC_local', 'pts_local', 'GF', 'GC'],
                axis=1,
                inplace=True)

        df_results = self.df[self.df['jornada'] == df.jornada.values[0]]
        df_results.loc[:, 'is_local'] = 1

        df = df.merge(df_results[['local', 'is_local']],
                      how='left',
                      left_on='team',
                      right_on='local').drop(['local'], axis=1)

        df = df.fillna(0)

        match = ['local', 'visitante']

        for t in match:

            other = [x for x in match if x != t][0]

            if t == 'local':
                df_temp = df[df['is_local'] == 1.0].copy()
            elif t == 'visitante':
                df_temp = df[df['is_local'] == 0.0].copy()
            else:
                raise ValueError('No dataframe to play with')

            df_temp.rename(columns={f'PG_{t}': 'PG',
                                    f'PP_{t}': 'PP',
                                    f'PE_{t}': 'PE'},
                           inplace=True)
            df_temp.drop([f'PG_{other}', f'PE_{other}', f'PP_{other}', 'pts', 'is_local'],
                         axis=1,
                         inplace=True)

            df_results = df_results.merge(df_temp,
                                          how='left',
                                          left_on=[f'{t}', 'jornada'],
                                          right_on=['team', 'jornada'])

         df_results.drop(['local', 'visitante', 'local_goals', 'visitante_goals',
                          'team_x', 'team_y', 'is_local'],
                         axis=1,
                         inplace=True)

        return df_results

    @staticmethod
    def clean_before_prediction(X_train, y_train, X_test, y_test):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return:
        """

        def transform(x):
            if x == 'W':
                return 3
            elif x == 'T':
                return 1
            else:
                return 0


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
        df_last['PJ_local'] = df_last['PG_local'] + df_last['PE_local'] + df_last['PP_local']
        df_last['GF_local'] = np.where(df_last['local'], df_last['GF'], 0)
        df_last['GC_local'] = np.where(df_last['local'], df_last['GC'], 0)
        df_last['pts_local'] = np.where(df_last['local'],
                                        df_last['score'],
                                        0)

        # visitante stats
        df_last['PG_visitante'] = np.where(df_last['winner'] == 2, np.where(~df_last['local'], 1, 0), 0)
        df_last['PE_visitante'] = np.where(df_last['winner'] == 0, np.where(~df_last['local'], 1, 0), 0)
        df_last['PP_visitante'] = np.where(df_last['winner'] == 1, np.where(~df_last['local'], 1, 0), 0)
        df_last['PJ_visitante'] = df_last['PG_visitante'] + df_last['PE_visitante'] + df_last['PP_visitante']
        df_last['GF_visitante'] = np.where(~df_last['local'], df_last['GF'], 0)
        df_last['GC_visitante'] = np.where(~df_last['local'], df_last['GC'], 0)
        df_last['pts_visitante'] = np.where(~df_last['local'],
                                            df_last['score'],
                                            0)

        # global stats
        df_last['PG'] = df_last['PG_local'] + df_last['PG_visitante']
        df_last['PE'] = df_last['PE_local'] + df_last['PE_visitante']
        df_last['PP'] = df_last['PP_local'] + df_last['PP_visitante']
        df_last['PJ'] = df_last['PG'] + df_last['PE'] + df_last['PP']
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

            df_last = df_last.merge(df_prev_jornada[['team', 'pts']],
                                    how='left',
                                    left_on='team',
                                    right_on='team')
            df_last.rename(columns={'pts_x': 'pts', 'pts_y': '1_match_ago_pts'}, inplace=True)

            for i in range(1, 7):
                if '{}_match_ago'.format(i) in df_prev_jornada.columns:
                    x = df_prev_jornada.copy()
                    x = x[['{}_match_ago'.format(i), 'team']]
                    x.rename(columns={'{}_match_ago'.format(i): '{}_match_ago'.format(i+1)},
                             inplace=True)
                    df_last = df_last.merge(x, how='left', on='team')

            df_last['1_match_ago'] = np.where(df_last['pts'] - df_last['1_match_ago_pts'] == 3,
                                              'W',
                                              np.where(df_last['pts'] - df_last['1_match_ago_pts'] == 1,
                                                       'T',
                                                       np.where(df_last['pts'] - df_last['1_match_ago_pts'] == 0,
                                                                'L',
                                                                'UNKNOWN'))
                                          )

            df_last.drop('1_match_ago_pts', axis=1, inplace=True)

        df_last = df_last.sort_values(by='pts', ascending=False).reset_index(drop=True)
        df_last = df_last.reset_index().rename(columns={'index': 'ranking'})
        df_last['ranking'] = df_last['ranking']+1

        return df_last
