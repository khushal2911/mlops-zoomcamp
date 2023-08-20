Model is trained on Jan'23 Divvy Tripdata
Validated on Feb'23 Divvy Tripdata
Scoring and Monitoring is done on Mar'23 Divvy Tripdata.

Experiments tracking results are saved in "mlruns.csv" file

Steps to train the model and track experiments
1. navigate to "training_tracking_registry/" directory in the project folder once project environment is setup.
2. Open mlflow server in one terminal =>
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
3. Run training code in another terminal =>
	python project_model_training.py
4. Register the best (lowest RMSE) model with 
	python register_model.py
