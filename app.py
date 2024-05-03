from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)

app = application


# Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file)
                print(df)
                print("Before Prediction")
                predict_pipeline = PredictPipeline()
                print("Mid Prediction")
                results = predict_pipeline.predict(df)
                print("after Prediction")
                if results[0] == 0:
                    output = "Non Seizure"
                else:
                    output = "Seizure"
                return render_template('home.html', results=output)

            except Exception as e:
                return str(e)


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080)
