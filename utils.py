import pandas as pd
import os
from sklearn.model_selection import train_test_split


def generate_dataset(rider_dir: str):
    dfs = []
    for d in os.listdir(rider_dir):
        path = f"{rider_dir}/{d}/processed_race_data.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[['team', 'dob', 'country', 'height', 'weight',
                     'one_day_races', 'gc', 'time_trial', 'sprint', 'climber', 'uci_world',
                     'pcs_ranking',
                     'distance', 'race_category', 'points_scale', 'uci_scale', 'parcours_type',
                     'profilescore', 'vert._meters', 'race_ranking',
                     'startlist_quality_score', 'stage_number', 'day', 'month',
                     'year', 'time_since_last_race', 'race_days_this_year',
                     'profile_score_vert_ratio', 'best_result_this_year',
                     'best_result_similar_races', 'result']]
            dfs.append(df)

    return pd.concat(dfs)


def test_train_validation_split(rider_dir: str):
    if not os.path.exists('data'):
        os.mkdir('data')

    for l in ['train', 'validation', 'test']:
        if not os.path.exists(f'data/{l}'):
            os.mkdir(f'data/{l}')

    data = generate_dataset(rider_dir)
    data = data.sort_values(by=['year', 'month', 'day'])
    train, test = train_test_split(data, train_size=0.8)
    train = data[:int(len(data) * 0.8)]
    nt = data[int(len(data) * 0.8):]
    val, test = train_test_split(nt, train_size=0.5, shuffle=True)

    train.to_csv('data/train/train.csv')
    val.to_csv('data/validation/val.csv')
    test.to_csv('data/test/test.csv')


def preprocess_categorical(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].value_counts().idxmax())
            df[col] = pd.get_dummies(df[col])
    return df

def preprocess_numerical(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].max())
    return df

def preprocess_all(df):
    return preprocess_categorical(preprocess_numerical(df))