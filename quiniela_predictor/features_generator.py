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

    file_path = '{k}{f}'.format(k=key,
                                f=filename)

    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
        logger.info('File {} created'.format(file_path))
    else:
        logger.warning('File {} already exists'.format(file_path))

    return


def main():

    today = dt.date.today()
    today_folder = '{y}/{m}/{d}/'.format(y=today.year,
                                         m=today.month,
                                         d=today.day)

    # read matches
    df_all = pd.read_csv(input_file)

    if df_all is None or len(df_all) == 0:
        raise ValueError('Datraframe is null or empty')

    for liga in ligas:

        logger.info('Generating data for competition: {}'.format(liga))

        df = df_all[df_all['liga'] == liga]
        df.drop('liga', axis=1, inplace=True)

        featex = Features(df)

        # create stats
        featex.jornada_generator()
        featex.local_visitante()
        featex.goles()
        featex.ganador()

        '''
        Build results dataset
        '''
        key = './files/{liga}/partit_a_partit/{today}'.format(liga=liga, today=today_folder)
        filename = 'partit_a_partit.csv'
        load_files(key, filename, featex.df)

        featex.team_by_team()

        df_prev_jornada = pd.DataFrame()
        df_predictor_dataset_old = pd.DataFrame()

        for jornada in sorted(featex.df_team.jornada.unique()):

            '''
            Build clasificacion dataset
            '''
            df_temp = featex.clasificacion(featex.df_team[featex.df_team['jornada'] == jornada],
                                           df_prev_jornada)

            if df_temp is not None:
                df_prev_jornada = df_temp.copy()
            else:
                logger.error('clasificacion returned None')
                break

            key = './files/{liga}/clasificacion/{today}'.format(liga=liga, today=today_folder)
            filename = 'clasificacion_{j}.csv'.format(j=jornada)
            load_files(key, filename, df_temp)

            '''
            Build match prediction dataset
            '''
            df_predictor_match_dataset = featex.predictor_dataset_match(df_temp)
            key = './files/{liga}/predictor_match_dataset/{today}'.format(liga=liga,
                                                                          today=today_folder)
            filename = 'predictor_match_dataset_{j}.csv'.format(j=jornada)
            load_files(key, filename, df_predictor_match_dataset)

            '''
            Build team prediction dataset
            '''
            df_predictor_dataset = featex.predictor_dataset_team_by_team(df_temp)

            key = './files/{liga}/predictor_dataset/{today}'.format(liga=liga,
                                                                    today=today_folder)
            filename = 'predictor_dataset_{j}.csv'.format(j=jornada)
            load_files(key, filename, df_predictor_dataset)

            '''
            Build team result prediction dataset
            '''
            if len(df_predictor_dataset_old) > 0 and '1_match_ago' in df_predictor_dataset.columns:
                y = df_predictor_dataset_old.merge(df_predictor_dataset[['team', '1_match_ago']],
                                                   how='left',
                                                   on='team').rename(columns={'1_match_ago': 'result'})
                key = './files/{liga}/predictor_dataset_result/{today}'.format(liga=liga,
                                                                               today=today_folder)
                filename = 'predictor_dataset_result_{j}.csv'.format(j=jornada)
                load_files(key, filename, y)

            df_predictor_dataset_old = df_predictor_dataset[['team', 'jornada']].copy()
