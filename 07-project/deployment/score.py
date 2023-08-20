import os
import sys
import zipfile, requests, io
from geopy.distance import distance
import pickle
from datetime import datetime
import pandas as pd
import mlflow

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from dateutil.relativedelta import relativedelta


def read_data(filename: str,input_file:str):
    zf = zipfile.ZipFile(io.BytesIO(requests.get(input_file).content))
    df = pd.read_csv(zf.open(filename+'.csv'))
    
    df.started_at = pd.to_datetime(df['started_at'],errors='coerce')
    df.ended_at = pd.to_datetime(df['ended_at'],errors='coerce')
    df = df.dropna()

    df['duration'] = df['ended_at'] - df['started_at']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1)]# less than a minute bike rides are outliers and hence removed
	
    df['trip_distance'] = df.apply(lambda row: distance((row['start_lat'],row['start_lng']),(row['end_lat'],row['end_lng'])).km,axis=1)
    
    categorical = ['start_station_id', 'end_station_id','rideable_type','member_casual']
    df[categorical] = df[categorical].astype(str)
    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['start_station_id', 'end_station_id','rideable_type','member_casual']
    df[categorical] = df[categorical].astype(str)
    
    df['SRC_DST'] = df['start_station_id'] + '_' + df['end_station_id']

    categorical = ['SRC_DST','rideable_type','member_casual']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    logged_model = f's3://mlflow-artifacts-store-remote-2/4/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['started_at'] = df['started_at']
    df_result['start_station_id'] = df['start_station_id']
    df_result['end_station_id'] = df['end_station_id']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_csv(output_file, index=False)


@task(name='Reading Data, Applying Model and Saving predictions')
def apply_model(filename,input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info(f'reading the data from {input_file}...')
    df = read_data(filename,input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    logger.info(f'applying the model...')
    y_pred = model.predict(dicts)

    logger.info(f'saving the result to {output_file}...')

    save_results(df, y_pred, run_id, output_file)
    return None


def get_paths(run_date, run_id):
    #prev_month = run_date - relativedelta(months=1)
    year = run_date.year
    month = run_date.month 

    filename = f'{year:04d}{month:02d}-divvy-tripdata'
    input_file = f'https://divvy-tripdata.s3.amazonaws.com/{filename}.zip'
    #input_file = f'/home/ubuntu/notebooks/07-project/data/{year:04d}{month:02d}-divvy-tripdata.zip'
    output_file = f'/home/ubuntu/notebooks/07-project/deployment/output/{year:04d}{month:02d}_{run_id}.csv'
    #output_file = f's3://mlops-zoomcamp-prefect-data/output/07-project/year={year:04d}/month={month:02d}/{run_id}.csv'

    return filename, input_file, output_file


@flow
def divvy_trip_prediction(
        run_id: str,
        run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    filename, input_file, output_file = get_paths(run_date, run_id)

    apply_model(filename=filename,
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )


def run():
    year = int(sys.argv[1]) # 2023
    month = int(sys.argv[2]) # 3

    run_id = sys.argv[3] # '307dee98dcf84aeba7b600a806dec2db' 

    divvy_trip_prediction(
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()




