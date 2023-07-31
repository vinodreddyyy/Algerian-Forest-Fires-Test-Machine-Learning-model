import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

application = Flask(__name__)
app=application

# it has to interact with the pickle files
ridge_model=pickle.load(open(r"python\Coding Ninjas\Lecture Machine Learning\2 Linear Regression\FWI prediction\models\ridge.pkl","rb"))
standard_scaler=pickle.load(open(r"python\Coding Ninjas\Lecture Machine Learning\2 Linear Regression\FWI prediction\models\scaler.pkl","rb"))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict():
    if(request.method=="POST"):
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(scaled_data)

        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run()