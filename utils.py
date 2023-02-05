
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np


def assign_y(df):
    df['podium'] = np.where(df['result'] <= 3, 1, 0)
    df = df.drop('result', axis=1)
    return df


def generate_dataset(rider_dir: str):
    dfs = []
    for d in os.listdir(rider_dir):
        path = f"{rider_dir}/{d}/processed_race_data.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)

            df = assign_y(df)
            dfs.append(df)

    return pd.concat(dfs)


def test_train_validation_split(rider_dir: str, train_size=0.8):
    if not os.path.exists('data'):
        os.mkdir('data')

    for l in ['train', 'validation', 'test']:
        if not os.path.exists(f'data/{l}'):
            os.mkdir(f'data/{l}')

    data = generate_dataset(rider_dir)
    data = data.sort_values(by=['year', 'month', 'day'])
    train, test = train_test_split(data, train_size=train_size)
    train = data[:int(len(data) * train_size)]
    nt = data[int(len(data) * train_size):]
    val, test = train_test_split(nt, train_size=0.5, shuffle=True)

    train.to_csv('data/train/train.csv', index=False)
    val.to_csv('data/validation/val.csv', index=False)
    test.to_csv('data/test/test.csv', index=False)


