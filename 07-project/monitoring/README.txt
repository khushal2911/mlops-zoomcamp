(base) ubuntu@ip-172-31-9-143:~/soft $ wget https://github.com/docker/compose/releases/download/v2.19.1/docker-compose-linux-x86_64 -O docker-compose
(base) ubuntu@ip-172-31-9-143:~/soft $ chmod +x docker-compose

Model performance is monitored for "202303-divvy-tripdata.csv" as given below.
This needs a different environment. create 'mlops-project' conda env from requirements.txt in this folder.
      $ conda create -n 'mlops-project' 
      $ conda activate 'mlops-project
      (mlops-project)$ pip install -r requirements.txt

Run the following line to monitor model performance on Mar'23 data.
      $ monitor.py 2023 3 307dee98dcf84aeba7b600a806dec2db

This will run prefect workflow with tasks 'prepare db', 'read data and make predictions', and 'calculate-metrics'
Data drift is computed in reference to predictions on '202302-divvy-tripdata.csv' saved in /ref_data/ folder

------------------ Running Grafana and Adminer on Web Browser ---------------
      $ docker-compose up --build 
Now, open grafana on web browser "https:\\localhost:3000" username/pwd - admin/Q1w2e3r4t5
// If it doesn't work, forward ports 3000, 8080, and 5432 in VScode terminal.

Watch for following verbose on terminal where docker-compose is running. (e.g. '05-monitoring' folder)
    05-monitoring-grafana-1  | logger=context userId=1 orgId=1 uname=admin t=2023-07-06T14:19:24.680689474Z
    level=info msg="Request Completed" method=GET path=/api/live/ws status=-1 
    remote_addr=172.20.0.1 time_ms=1 duration=1.198579ms size=0 referer= handler=/api/live/ws

// Enter following details on Adminer page opened in the browser
System - PostgreSQL
Server - 172.20.0.1 (remote_addr from above message in terminal)
Username - postgres
Password - example
Database - test
