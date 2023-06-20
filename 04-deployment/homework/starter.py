#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def run():
    year = int(sys.argv[1]) # 2022
    month = int(sys.argv[2]) # 3

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    average_preds = y_pred.mean()
    #df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    #df['y_pred'] = y_pred
    
    return print(average_preds)

if __name__ == '__main__':
    run()
