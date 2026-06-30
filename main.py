from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("xgb_model.pkl")

# Define your input features here
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # add all your columns...

@app.get("/")
def home():
    return {"status": "Model API is running!"}

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}