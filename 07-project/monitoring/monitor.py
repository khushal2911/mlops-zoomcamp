import datetime
import time
import random
import logging 
import pandas as pd
import io
import sys
import requests, zipfile
from geopy.distance import distance
import mlflow
import psycopg
import warnings
warnings.filterwarnings('ignore')

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists monitoring_metrics;
create table monitoring_metrics(
	timestamp timestamp,
	prediction_drift float,
	median_trip_distance float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

reference_data = pd.read_csv('ref_data/202302-divvy-tripdata_reference.csv')

def load_model(run_id):
    logged_model = f's3://mlflow-artifacts-store-remote-2/4/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


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

num_features = ['trip_distance']
cat_features = ['SRC_DST','rideable_type','member_casual']
column_mapping = ColumnMapping(
    target=None,
    prediction='predicted_duration',
    numerical_features=num_features,
    categorical_features=cat_features
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='predicted_duration'),
    ColumnQuantileMetric(column_name='trip_distance',quantile=0.5),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task(retries=2, retry_delay_seconds=5, name="prepare db")
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task(retries=2, retry_delay_seconds=5, name="read data and make predictions")
def process(year: int, month: int, run_id: str):
	filename = f'{year:04d}{month:02d}-divvy-tripdata'
	input_file = f'https://divvy-tripdata.s3.amazonaws.com/{filename}.zip'
	
	model = load_model(run_id)
	curr_data = read_data(filename,input_file)
	curr_data['predicted_duration'] = model.predict(prepare_dictionaries(curr_data))
	return curr_data

	
@task(retries=2, retry_delay_seconds=5, name="calculate metrics")
def calculate_metrics_postgresql(begin,curr_data,curr, i):
	current_data = curr_data[(curr_data.started_at >= (begin + datetime.timedelta(i))) &
		(curr_data.started_at < (begin + datetime.timedelta(i + 1)))]

	report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)

	result = report.as_dict()

	prediction_drift = result['metrics'][0]['result']['drift_score']
	median_trip_distance = result['metrics'][1]['result']['current']['value']
	num_drifted_columns = result['metrics'][2]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][3]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into monitoring_metrics(timestamp, prediction_drift, median_trip_distance, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, median_trip_distance, num_drifted_columns, share_missing_values)
	)

@flow
def run(): #batch-monitoring-backfill
	prep_db()
	year = int(sys.argv[1]) # 2023
	month = int(sys.argv[2]) # 3
	run_id = sys.argv[3] # '307dee98dcf84aeba7b600a806dec2db' 
	begin = datetime.datetime(year, month, 1, 0, 0)
	end = datetime.datetime(year,month+1,1,0,0)
	days = (end-begin).days
	curr_data = process(year=year, month=month, run_id=run_id)
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
		for i in range(0, days):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(begin,curr_data,curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	run() #batch-monitoring-backfill
