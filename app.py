import pickle
import flask
from flask import app,request,Flask,Request,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import sklearn

app=Flask(__name__)
pickled_model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json["single_input"] ## single_input is key which is there in postman
    new_2D_data=[list(data.values())]
    output=pickled_model.predict(new_2D_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    new_2D_data=[np.array(data)]
    output=pickled_model.predict(new_2D_data)[0]
    return render_template('home.html',prediction_text="Airfoil Pressure is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)
