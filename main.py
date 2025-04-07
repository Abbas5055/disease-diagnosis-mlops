from fastapi import FastAPI
import joblib
import pandas as pd
app = FastAPI()
model = joblib.load("model.pkl")
@app.get("/")
def home():
    return {"message": "Welcome to Disease Prediction API!"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
        return {"prediction": int(prediction)}
