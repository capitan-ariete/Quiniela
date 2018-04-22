import os
import logging.config
from logging.config import fileConfig
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# TODO esto tiene que ser env variable
input_file = './files/clasificacion/2018/4/22/clasificacion_33.csv'


def main():
    df = pd.read_csv(input_file)

    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative = 2, n_redundant = 0,
                               random_state = 0, shuffle = False)

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf.fit(X, y)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=2, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                           oob_score=False, random_state=0, verbose=0, warm_start=False)

    print(clf.feature_importances_)


if __name__ == '__main__':
    logger.info("It's show time.")
    main()
    logger.info("You have been terminated.")
