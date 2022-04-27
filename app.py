# Importing essential libraries and modules

import pickle

import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request

import config

# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

crop_yeild_pridiction_path = 'models/DecisionTree.pkl'
crop_yeild_pridiction_model = pickle.load(
    open(crop_yeild_pridiction_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)


# render home page

@app.route('/')
def home():
    title = 'Home'
    return render_template('index.html', title=title)


@app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop Recommendation'
    return render_template('crop.html', title=title)


@app.route('/working')
def working():
    title = 'Working-Procedure'
    return render_template('working-procedure.html', title=title)


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Crop-yield Prediction'

    return render_template('fertilizer.html', title=title)


# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature= float(request.form['temperature'])
        humidity = float(request.form['humidity'])



        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, title=title)


@app.route('/yield-predict', methods=['POST'])
def yield_prediction():
    title = 'Yield_prediction'

    if request.method == 'POST':
        crop_name = str(request.form['cropname'])
        country_name = str(request.form['countryname'])
        rainfall = float(request.form['rainfall'])
        pesticide = float(request.form['pesticides'])
        temp = float(request.form['avg_temp'])

        yield_df = pd.read_csv('Data/yield_df.csv')

        yield_df1 = yield_df[yield_df.Area == country_name]

        final_data = yield_df1[yield_df1.Item == crop_name]

        # data_after_one_hot_encode = pd.get_dummies(final_data, columns=['Area', "Item"], prefix=['Country', "Item"])
        # features = data_after_one_hot_encode.loc[:, data_after_one_hot_encode.columns != 'hg/ha_yield']
        # features = features.drop(['Year'], axis=1)
        #
        # # data = np.array([[crop_name, country_name, rainfall, pesticide, temp]])
        # my_prediction = crop_yeild_pridiction_model.predict(features[1:2])

        my_prediction = final_data['hg/ha_yield'].iloc[0]
        final_prediction = my_prediction

        return render_template('fertilizer-result.html', prediction=final_prediction, title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
