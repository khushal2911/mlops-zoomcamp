navigate to "orchestration/" directory.
In one terminal execute following commands
	prefect project init
	prefect server start
Open http://127.0.0.1:4200 in browser to view Prefect server UI.

In second terminal execute following command
	prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
	python orchestrate.py
This will execute the prefect workflows 'reading data files', 'Adding model features-0', 'train_best_model-0' and
'email_send_message-0'. In the end I receive email notification when all the flows are successful.

"orchestrate_s3.py" stored artifacts on s3 bucket with following run_id for the best-rmse model.
run_id - 307dee98dcf84aeba7b600a806dec2db
s3_path - "s3://mlflow-artifacts-store-remote-2/4/{run_id}/artifacts/model"

'''
Created the deployment named 'mlops-project-orch' with 'main_flow' as entrypoint and 'mlopsprojectpool' as processworker.
'''
	prefect deploy orchestrate.py:main_flow --name mlops-project-orch --pool mlopsprojectpool --work-queue ml
	prefect worker start -p mlopsprojectpool 

'''
Email Notification Block is created to send an automated email to my personal email ID when prefect workflow is 
successfully completed.