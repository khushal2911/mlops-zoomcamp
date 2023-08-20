Steps to train the model and track experiments
1. navigate to "training_tracking_registry/" directory in the project folder
2. Open mlflow server in one terminal =>
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
3. Run training code in another terminal=>
	python project_model_training.py
4. Register the best (lowest RMSE) model =>
	python register_model.py