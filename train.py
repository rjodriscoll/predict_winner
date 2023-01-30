import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

# Load the data
data = pd.read_csv("data/train/train.csv")


X, y = data.drop('podium', axis = 1), data[['podium']]
# Define the columns for numerical and categorical features
num_cols = [col for col in X.select_dtypes(include=["int64", "float64"]).columns if col not in ['podium', 'result']]
cat_cols = X.select_dtypes(include=["object"]).columns

pipeline = Pipeline([
    ('ct', ColumnTransformer([
        ('num', SimpleImputer(strategy='max'), num_cols),
        ('cat', OneHotEncoder(), cat_cols),
    ])),
    ('fs', SelectKBest(f_classif, k=10)),
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

with open("models/pipeline.pkl", "rb") as file:
    pipeline = pickle.dump(pipeline, file)
