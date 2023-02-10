from joblib import dump
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from utils import CustomProcess
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper


train = pd.read_csv("data/train/train.csv")

X, y = train.drop('podium', axis = 1), train[['podium']]


cols_to_select = ['team', 'country', 'one_day_races', 'gc', 'time_trial', 'sprint',
 'climber', 'all_time', 'uci_world', 'pcs_ranking', 'date_scraped',
 'distance', 'date', 'start_time', 'race_category', 'points_scale', 'uci_scale',
  'parcours_type',
 'profilescore', 'vert._meters', 'race_ranking',
 'startlist_quality_score', 'stage_number', 'day', 'month', 'year', 
 'time_since_last_race', 'race_days_this_year',
 'profile_score_vert_ratio', 'best_result_this_year',
 'best_result_similar_races']
X, y = train[cols_to_select], train[['podium']]

num_cols = [col for col in X.select_dtypes(include=["int64", "float64"]).columns if col not in ['podium', 'result']]
cat_cols = X.select_dtypes(include=["object"]).columns

pipeline = Pipeline([
    ('mapper', DataFrameMapper([
        (num_cols, SimpleImputer(strategy='most_frequent')),
        (cat_cols, OneHotEncoder())
    ])),
    ('xgb', RandomizedSearchCV(
        XGBClassifier(), 
        param_distributions={
            'learning_rate': [0.01, 0.1, 0.5],
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 5, 7],
        },
        n_iter=10,
    )),

])
pipeline.fit(X, y)



dump(pipeline, 'models/pipeline.joblib')

val_data = pd.read_csv("data/validation/val.csv")

X_val, y_val = val_data.drop('podium', axis = 1), val_data[['podium']]

scores = cross_val_score(pipeline, X_val, y_val, cv=3)

print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
