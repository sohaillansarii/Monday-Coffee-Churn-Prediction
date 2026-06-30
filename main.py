from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")
model = joblib.load("xgb_model.pkl")

class CustomerData(BaseModel):
    account_age_months: float
    avg_order_value: float
    total_orders: float
    discount_usage_rate: float
    return_rate: float
    customer_support_tickets: float
    loyalty_member: int        # 0 or 1
    browsing_frequency_per_week: float
    cart_abandonment_rate: float
    product_review_score_avg: float
    engagement_score: float
    satisfaction_score: float
    price_sensitivity_index: float
    total_spend: float

@app.get("/")
def home():
    return {"status": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_label": "Will Churn" if prediction == 1 else "Will NOT Churn",
        "churn_probability": round(float(probability), 4)
    }
