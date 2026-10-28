from fastapi import FastAPI
from app.schema import LoanRequest
import numpy as np
import joblib

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0"
)

model = joblib.load("model/credit_risk_model.joblib")


@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is running!"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    features = np.array([[
        data.person_age,
        data.person_income,
        data.person_home_ownership,
        data.person_emp_length,
        data.loan_intent,
        data.loan_grade,
        data.loan_amnt,
        data.loan_int_rate,
        data.loan_percent_income,
        data.cb_person_default_on_file,
        data.cb_person_cred_hist_length
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        "prediction": int(prediction),
        "probability_default": float(probability[1]),
        "probability_safe": float(probability[0])
    }