
from flask import Flask, request, jsonify, render_template
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = load_model('model01')
cols=['Age','BusinessTravel', 'DailyRate', 'Department',
       'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement',
       'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
       'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'TotalWorkingYears', 'YearsInCurrentRole']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [x for x in request.form.values()]
    print("x_test")
    print(x_test)
    print(" ")

    final=np.array(x_test)
    print("final")
    print(final)
    print(" ")

    data=pd.DataFrame([final],columns=cols)
    print("data")
    print(data)
    print(" ")

    prediction = predict_model(model,data=data,round=0)
    print("prediction")
    print(prediction)
    print(" ")

    output=int(prediction['Label'][0])
    return render_template('index.html',
  prediction_text=
  'employees Attrition is {}(0 means Good,1 means Bad )'.format(output))
if __name__ == "__main__":
    app.run(debug=True)
