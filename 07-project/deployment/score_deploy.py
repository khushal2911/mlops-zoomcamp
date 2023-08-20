from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from score import divvy_trip_prediction
from datetime import datetime

deployment = Deployment.build_from_flow(
    flow=divvy_trip_prediction,
    name="divvy_trip_prediction",
    parameters={
        "run_id": "307dee98dcf84aeba7b600a806dec2db",
        "run_date" : datetime(year=2023,month=3,day=1)
    },
    schedule=CronSchedule(cron="0 3 2 * *"),
    work_pool_name ="mlopsprojectpool",
    work_queue_name="ml",
)

deployment.apply()
