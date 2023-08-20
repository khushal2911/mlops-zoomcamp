ML Project is described in detail in "Project Description.txt"

Create a pipenv and install all the dependencies from .txt file
1. 	pipenv install -r requirements.txt
	pipenv shell
2. Directory structure
	i. data/ - contains zipped data files for the model training and validation.
		   They can also be downloaded from "https://divvy-tripdata.s3.amazonaws.com/index.html"
		   Model is trained on "202301-divvy-tripdata.csv" and validated on "202302-divvy-tripdata.csv".
	ii. training_tracking_registry/ - it consists of "project_model_training.py" and model "register_model.py "codes.
		   Also contains snapshots from MLflow server UI and .csv file listing all the experiment runs.
	iii. orchestration/ - Here, Model with The Best parameters is trained with fully deployed workflow 'main_flow'.
		   Training is also tracked using mlflow, and artifacts are stored under "mlruns/" folder.
		   Best Model run_id = '307dee98dcf84aeba7b600a806dec2db'
		   It has s3 version as well, where artifacts are stored in my s3 bucket "mlflow-artifact-store-remote-2".
	iv. deployment/ - Model deployment is done in both "batch" mode and "web-service" mode locally and with docker.
			It can be tested by running "test.py" with various inputs. Check the folder for more details. 
	v. monitoring/ - Contains monitoring codes using which model performance can be tracked on any month's data.
			Reference Data with predictions for 2023-02 is saved inside "ref_data/" folder.
			Added screenshots for Dashboard Monitoring in Graphana for 2023-03, along with its Prefect Workflow.
			Data stored in postgreSQL on Adminer is saved in "Mar23 monitoring_metrics_adminer.csv"
3. There is a separate README file in each project directory for further understanding and evaluations.
4. I'm yet to wrap my head around 'Best Engineering Practices'. Will include them in future.
