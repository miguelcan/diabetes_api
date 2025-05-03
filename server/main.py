from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from classifier.model import DiabetesModel
import logging

logging.basicConfig(filename="app.log", level=logging.INFO, format="%(levelname)s:%(message)s")
app = FastAPI()
model = DiabetesModel("models/diabetes_model.pkl")

class Features(BaseModel):
    data: list

@app.post("/predict")
def predict(features: Features):
    if len(features.data) != 10:
        raise HTTPException(status_code=400, detail="Input data must contain exactly 10 values.")

    logging.info(f"Received data: {features.data}")
    prediction = model.predict(features.data)
    return {"prediction": prediction}
