import os
import pickle
import click
import mlflow

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

TRACKING_SERVER_HOST = 'localhost' 
EXPERIMENT_NAME = "chicago-divvy-trip-prediction-experiment"

client = MlflowClient(f"http://{TRACKING_SERVER_HOST}:5000")

def run_register_model():

    client = MlflowClient()

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        max_results=1,
        order_by=["metrics.rmse ASC"]
        )[0].info.run_id
    
    mlflow.register_model(
        model_uri=f"runs:/{best_run}/models",
        name='Chicago-Divvy-Trip-Predictor-BestRMSE-XGBoost'
    )
    print('{} run id model registered successfully'.format(best_run))

if __name__ == '__main__':
    run_register_model()
