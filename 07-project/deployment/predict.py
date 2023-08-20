import os
import pickle

import mlflow
from flask import Flask, request, jsonify

# export RUN_ID=307dee98dcf84aeba7b600a806dec2db in terminal before running 'python predict.py' and test.py
RUN_ID = os.getenv('RUN_ID')
 
logged_model = f's3://mlflow-artifacts-store-remote-2/4/{RUN_ID}/artifacts/model'
#logged_model = f'/model/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['SRC_DST'] = '%s_%s' % (ride['start_station_id'], ride['end_station_id'])
    features['rideable_type'] = ride['rideable_type']
    features['member_casual'] = ride['member_casual']
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('chicago-divvy-bikes-trip-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
