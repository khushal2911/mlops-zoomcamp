navigate to "orchestration/" directory start prefect server
	prefect project init
	prefect server start
Open another terminal to set prefect api and run the program
	prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
	python orchestrate.py

Prefect server UI can be viewed at http://127.0.0.1:4200 in browser.


This will execute following prefect workflow and tasks.
	'reading data files'--> 'Adding model features-0' --> 'train_best_model-0' --> 'email_send_message-0'. 
Email Notification Block is created to send an automated email to my personal email ID when prefect workflow is 
successfully completed.

"orchestrate_s3.py" stored artifacts on s3 bucket with following run_id for the best-rmse model.
run_id - 307dee98dcf84aeba7b600a806dec2db
s3_path - "s3://mlflow-artifacts-store-remote-2/4/{run_id}/artifacts/model"

Created the deployment named 'mlops-project-orch' with 'main_flow' as entrypoint and 'mlopsprojectpool' as processworker.
	prefect deploy orchestrate.py:main_flow --name mlops-project-orch --pool mlopsprojectpool --work-queue ml
	prefect worker start -p mlopsprojectpool 

