from flask import Flask, request, Response
from rossmann import Rossmann
import pickle
import pandas as pd
import os

# loading model - pipeline
model = pickle.load(open('webapp/model/pipe.pkl', 'rb'))


# initialize app
app = Flask( __name__ )

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Rossmann API!"


@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instantiate rossmann
        initial_pipeline = Rossmann()

        # data cleaning
        data1 = initial_pipeline.data_cleaning(test_raw)

        # feature eng
        data2 = initial_pipeline.feature_engineering(data1)

        # data prep
        data3 = initial_pipeline.data_prep(data2)

        # prediction
        data_response = initial_pipeline.get_prediction(model, test_raw, data3)

        # create response json
        response_dict = {'data_response': data_response, 'raw_data': test_json}
        response_json = pd.io.json.dumps(response_dict)

        return Response(response_json, status=200, mimetype='application/json')

    else:
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)