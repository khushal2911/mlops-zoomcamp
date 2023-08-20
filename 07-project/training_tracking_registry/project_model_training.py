import pickle
import pandas as pd
from zipfile import ZipFile
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from geopy.distance import distance

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


# MLflow settings
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("chicago-divvy-trip-prediction-experiment")

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_data(filename: str):
    data_path = '/home/ubuntu/notebooks/07-project/data'
    zip_file = ZipFile(f'{data_path}/{filename}.zip')
    df = pd.read_csv(zip_file.open(f'{filename}.csv'))
    
    df.started_at = pd.to_datetime(df['started_at'],errors='coerce')
    df.ended_at = pd.to_datetime(df['ended_at'],errors='coerce')
    df = df.dropna()

    df['duration'] = df['ended_at'] - df['started_at']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1)]# less than a minute rides are removed
	
    df['trip_distance'] = df.apply(lambda row: distance((row['start_lat'],row['start_lng']),(row['end_lat'],row['end_lng'])).km,axis=1)
    
    categorical = ['start_station_id', 'end_station_id','rideable_type','member_casual']
    df[categorical] = df[categorical].astype(str)

    return df


def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    df_train['SRC_DST'] = df_train['start_station_id'] + '_' + df_train['end_station_id']	
    df_val['SRC_DST'] = df_val['start_station_id'] + '_' + df_val['end_station_id']

    categorical = ['SRC_DST','rideable_type','member_casual']
    numerical = ['trip_distance']
    
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
	
    return X_train, X_val, y_train, y_val, dv


train_path = "202301-divvy-tripdata"
val_path = "202302-divvy-tripdata"

df_train = read_data(train_path)
df_val = read_data(val_path)

X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=500,
            evals=[(valid, 'validation')],
            early_stopping_rounds=25
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}


search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:squarederror',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials(),
    return_argmin=False
)

print('max_depth: {}'.format(best_result['max_depth']))
print('learning_rate: {}'.format(best_result['learning_rate']))
print('reg_alpha: {}'.format(best_result['reg_alpha']))
print('reg_lambda: {}'.format(best_result['reg_lambda']))
print('min_child_weight: {}'.format(best_result['min_child_weight']))










