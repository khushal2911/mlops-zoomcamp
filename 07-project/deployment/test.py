import requests, sys

def run():
    ride = {
    'start_station_id' : sys.argv[1],
    'end_station_id' : sys.argv[2],
    'rideable_type' : sys.argv[3] , # 'electric' or 'classic'
    'member_casual' : str(sys.argv[4]), # 'member' or 'casual'
    'trip_distance' : float(sys.argv[5]) # distance in kms
    }
    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=ride)
    print(response.json())


if __name__ == '__main__':
    run()
