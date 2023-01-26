
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np


def assign_y(df):
    df['podium'] = np.where(df['result'] <= 3, 1, 0)
    df = df.drop('result', axis = 1)
    return df


def generate_dataset(rider_dir: str):
    dfs = []
    for d in os.listdir(rider_dir):
        path = f"{rider_dir}/{d}/processed_race_data.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            df = preprocess_all(df)
            df= assign_y(df)
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

    train.to_csv('data/train/train.csv', index=False)
    val.to_csv('data/validation/val.csv', index=False)
    test.to_csv('data/test/test.csv', index=False)


def preprocess_categorical(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].value_counts().idxmax())
            df = pd.concat([df, pd.get_dummies(
                df[col], prefix=col, prefix_sep='_', dummy_na=False, columns=None)], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df


def preprocess_numerical(df):
    df = df.copy()
    for col in [col for col in df.columns if col not in ['podium', 'result']]:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].max())
    return df


def preprocess_all(df):
    df= preprocess_categorical(preprocess_numerical(df))
    return df 