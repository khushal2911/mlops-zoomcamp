
import pandas as pd
from zipfile import ZipFile
from geopy.distance import distance
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task
from prefect_aws import S3Bucket
from prefect.artifacts import create_markdown_artifact
from datetime import date
from prefect_email import EmailServerCredentials, email_send_message
from prefect import runtime

TRACKING_SERVER_HOST = "ec2-16-171-165-47.eu-north-1.compute.amazonaws.com"

@task(name='reading data files',retries=3, retry_delay_seconds=2)
def read_data(filename: str):
    data_path = "/home/ubuntu/notebooks/07-project/data"
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

@task(name='Adding model features')
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple(
    [
        dict,
        dict,
        np.ndarray,
        np.ndarray,
    ]
):
    df_train['SRC_DST'] = df_train['start_station_id'] + '_' + df_train['end_station_id']	
    df_val['SRC_DST'] = df_val['start_station_id'] + '_' + df_val['end_station_id']

    categorical = ['SRC_DST','rideable_type','member_casual']
    numerical = ['trip_distance']
    
    X_train = df_train[categorical + numerical].to_dict(orient="records")
    X_val = df_val[categorical + numerical].to_dict(orient="records")
    
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
	
    return X_train, X_val, y_train, y_val

@task(name='training the best model',log_prints=True)
def train_best_model(
    X_train: dict,
    X_val: dict,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():

        best_params = {
            "learning_rate": 0.09002710203069089,
            "max_depth": 4,
            "min_child_weight": 7.370049044160041,
            "objective": "reg:squarederror",
            "reg_alpha": 0.007060723631897894,
            "reg_lambda": 0.11278036580451371,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        pipeline = make_pipeline(DictVectorizer(),xgb.XGBRegressor(**best_params))
        
        pipeline.fit(X_train,y_train)

        y_pred = pipeline.predict(X_val)
                                 
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        markdown__rmse_report = f"""# RMSE Report

        ## Summary

        Duration Prediction 

        ## RMSE XGBoost Model

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse:.2f} |
        """

        create_markdown_artifact(
            key="duration-model-report", markdown=markdown__rmse_report
        )

    return None

@flow
def send_email_notification_flow():
    flow_run_name = runtime.flow_run.name
    email_credentials_block = EmailServerCredentials.load("email-server-creds") 
    email_send_message(
        email_server_credentials=email_credentials_block,
        subject=f"Flow run {flow_run_name!r} success",
        msg=f"Flow run {flow_run_name!r} succeeded and notified through email.",
        email_to=email_credentials_block.username,
    )


@flow
def main_flow_s3(
    train_path: str = "202301-divvy-tripdata",
    val_path: str = "202302-divvy-tripdata",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("chicago-divvy-trip-prediction-with-prefect")

    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Preparing Features
    X_train, X_val, y_train, y_val = add_features(df_train, df_val)
    # Transform and Train
    train_best_model(X_train, X_val, y_train, y_val)
    send_email_notification_flow()

if __name__ == "__main__":
    main_flow_s3()
