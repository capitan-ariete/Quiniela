"""Refs
https://github.com/neerajnj10/soccer-analytics-with-python-mongoDB-and-R/blob/master/FinalReport.md
https://towardsdatascience.com/o-jogo-bonito-predicting-the-premier-league-with-a-random-model-1b02fa3a7e5a
https://github.com/tuangauss/Various-projects/blob/master/R/EPL/sim.R
"""

import os

import logging.config
import numpy as np
import pandas as pd
import datetime as dt

from features import Features

logger = logging.getLogger(__name__)

# input_file = './files/file.csv'
input_file = './files/file_many_seassons.csv'
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
        raise ValueError('clasificacion returned None')

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


# TODO to develop
# TODO necesito ampliar a 5 seassons para tener mejores resultados!!
def poisson_prediction(df):
    import operator
    # TODO next var
    # next_journey_matches # list of tuples of teams (home, away) [(barça, madrid), (bilbao, osasuna), ...]
    next_journey_matches = [('R. Sociedad', 'Barcelona'), ('Athletic', 'Real Madrid'), ('Valencia', 'Betis')] # TODO this is a toy
    table = dict()
    for match in next_journey_matches:
        sim_num = 10000
        result = {
            'prob_home_win': 0,
            'prob_tie': 0,
            'prob_away_win': 0
        }
        for _ in range(sim_num):
            if len(df_temp) > 3:
                mask = (
                        (df['local'] == match[0])
                        &
                        (df['visitante'] == match[1])
                )
                df_temp = df[mask].copy()
                avg_home_scored = round(df_temp.local_goals.mean(), 2)
                avg_away_scored = round(df_temp.visitante_goals.mean(), 2)
                h_goals = np.random.poisson(1, int(avg_home_scored))
                a_goals = np.random.poisson(1, int(avg_away_scored))
            else:
# TODO     h_scored = rpois(1, 1/2 * (ave[ave$Team == home,]$ave_scored_h + # ave[ave$Team == away,]$ave_conceded_a))
                h_goals = np.random.poisson(1, int(avg_home_scored))
                a_goals = np.random.poisson(1, int(avg_away_scored))

            if h_goals > a_goals:
                result['prob_home_win'] += 1
            elif h_goals == a_goals:
                result['prob_tie'] += 1
            else:
                result['prob_away_win'] += 1

        print(match, result)
        table[match] = max(result.items(), key=operator.itemgetter(1))[0]
    return table


# TODO pending filter by seassons
def process_league(league, df):
    """
    Process every single match for a given league.
    :warning: Working for La Liga and 2da division
    :param league:
    :param df:
    :return:
    """
    # prepare league dataset
    df = df[df['liga'] == league]
    df.drop('liga', axis=1, inplace=True)

    # invoke Features class
    features = Features(df)

    # create stats
    features.seasson()
    features.jornada_generator()
    features.local_visitante()
    features.goles()
    features.ganador()

    if 'seasson' not in features.df.columns:
        # :warning: this is the old process
        build_results()
    else:
        # TODO aqui ya puedo aplicar la solucion del post https://towardsdatascience.com/o-jogo-bonito-predicting-the-premier-league-with-a-random-model-1b02fa3a7e5a
        poisson_prediction(features.df)


def features_generator_orchestrator():
    """
    Features Orchestrator
    :return:
    """
    # read matches
    df_all = pd.read_csv(input_file)

    if df_all is None or len(df_all) == 0:
        raise ValueError('Input dataset is null or empty')

    for liga in ligas:
        logger.info(f'Generating data for competition: {liga}')
        process_league(league=liga, df=df_all)
        logger.info(f'League {liga} | prediction completed')
