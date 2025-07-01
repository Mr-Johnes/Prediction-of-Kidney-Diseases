import pandas as pd
from flask import Flask, request, render_template
import pickle
import numpy as np  # This import is missing in the original code but needed

app = Flask(__name__)  # initializing a flask app
model = pickle.load(open('CKD.pkl', 'rb'))  # loading the model

@app.route('/')  # route to display the home page
def home():
    return render_template('home.html')  # rendering the home page

@app.route('/Prediction', methods=['POST', 'GET'])  # route to display prediction page
def prediction():
    return render_template('indexnew.html')

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def predict():
    # reading the inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['red_blood_cells', 'pus_cell', 'blood_glucose_random', 'blood_urea','pedal_edema', 'anemia', 'diabetesmellitus', 'coronary_artery_disease']

    df = pd.DataFrame(features_value, columns=features_name)

    # predictions using the loaded model file
    output = model.predict(df)

    # showing the prediction results in a UI
    return render_template('result.html', prediction_text=str(output))

if __name__ == '__main__':
    app.run(debug=True)  # running the app
