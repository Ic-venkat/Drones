from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__,static_url_path='/static')

# Load the trained model


import json

@app.template_filter('parse_json')
def parse_json_filter(json_str):
    return json.loads(json_str)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/drone_database')
def drone_database():
    return render_template('drone_database.html')

@app.route('/drone_home')
def drone_home():
    return render_template('index.html')

@app.route('/drone_prediction')
def drone_prediction():
    return render_template('drone_pred.html')

@app.route('/drone_about')
def drone_about():
    return render_template('about.html')

@app.route('/drone_contact')
def drone_contact():
    return render_template('contact.html')

 
@app.route('/drone_login')
def drone_login():
    return render_template('login.html')

@app.route('/drone_signup')
def drone_signup():
    return render_template('signup.html')

rf_model = joblib.load('random_forest_model.joblib')
lr_model = joblib.load('logistic_regression_model.joblib')
svm_model = joblib.load('support_vector_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    flight_radius = float(request.form['flight_radius'])
    payload_height_in_meters = float(request.form['flight_height_in_meters'])
    range_distance = float(request.form['operations_per_day'])
    wind_speed = float(request.form['wind_speed'])
    budget = float(request.form['budget'])
    payload_capacity = float(request.form['payload_capacity'])
    camera_quality = float(request.form['camera_quality'])
   
    # Create a feature vector
    features = [[flight_radius, payload_height_in_meters, range_distance, wind_speed, budget, payload_capacity, camera_quality]]

    # Make predictions using the loaded models
    rf_prediction = rf_model.predict(features)[0]
    lr_prediction = lr_model.predict(features)[0]
    svm_prediction = svm_model.predict(features)[0]

    # Read the drone dataset
    df = pd.read_csv('drone_dataset.csv')

    # Get the data for each predicted drone
    rf_data = df[df['Drone Name'] == rf_prediction]
    lr_data = df[df['Drone Name'] == lr_prediction]
    svm_data = df[df['Drone Name'] == svm_prediction]

    # Reset the indices and convert to JSON
    rf_data.reset_index(drop=True, inplace=True)
    lr_data.reset_index(drop=True, inplace=True)
    svm_data.reset_index(drop=True, inplace=True)

    rf_data_json = rf_data.to_json(orient='records')
    lr_data_json = lr_data.to_json(orient='records')
    svm_data_json = svm_data.to_json(orient='records')

    rf_data_dict = json.loads(rf_data_json)[0]
    lr_data_dict = json.loads(lr_data_json)[0]
    svm_data_dict = json.loads(svm_data_json)[0]

    return render_template('result.html',
                           rf_drone_name=rf_data_dict['Drone Name'],
                           rf_flight_radius=rf_data_dict['Flight radius (KM)'],
                           rf_flight_height=rf_data_dict['Flight height(meters)'],
                           rf_operations_per_day=rf_data_dict['Operations per day'],
                           rf_wind_speed=rf_data_dict['Wind speed'],
                           rf_budget=rf_data_dict['Budget (EUR)'],
                           rf_payload_capacity=rf_data_dict['payloadCapacity'],
                           rf_camera_quality=rf_data_dict['cameraQuality'],
                           lr_drone_name=lr_data_dict['Drone Name'],
                           lr_flight_radius=lr_data_dict['Flight radius (KM)'],
                           lr_flight_height=lr_data_dict['Flight height(meters)'],
                           lr_operations_per_day=lr_data_dict['Operations per day'],
                           lr_wind_speed=lr_data_dict['Wind speed'],
                           lr_budget=lr_data_dict['Budget (EUR)'],
                           lr_payload_capacity=lr_data_dict['payloadCapacity'],
                           lr_camera_quality=lr_data_dict['cameraQuality'],
                           svm_drone_name=svm_data_dict['Drone Name'],
                           svm_flight_radius=svm_data_dict['Flight radius (KM)'],
                           svm_flight_height=svm_data_dict['Flight height(meters)'],
                           svm_operations_per_day=svm_data_dict['Operations per day'],
                           svm_wind_speed=svm_data_dict['Wind speed'],
                           svm_budget=svm_data_dict['Budget (EUR)'],
                           svm_payload_capacity=svm_data_dict['payloadCapacity'],
                           svm_camera_quality=svm_data_dict['cameraQuality']
                           )

if __name__ == '__main__':
    app.run(debug=True)
