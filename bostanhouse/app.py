from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            crim=float(request.form.get('crim')),
            zn=float(request.form.get('zn')),
            indus=float(request.form.get('indus')),
            chas=int(request.form.get('chas')),
            nox=float(request.form.get('nox')),
            rm=float(request.form.get('rm')),
            age=float(request.form.get('age')),
            dis=float(request.form.get('dis')),
            rad=int(request.form.get('rad')),
            tax=float(request.form.get('tax')),
            ptratio=float(request.form.get('ptratio')),
            b=float(request.form.get('b')),
            lstat=float(request.form.get('lstat'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("mid Prediction")
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)        
