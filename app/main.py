from fastapi import FastAPI
import requests
from pydantic import BaseModel
import logging

logging.basicConfig(filename="/app/app.log", level=logging.INFO, format="%(levelname)s:%(message)s")
app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/classify")
def classify(data: InputData):
    logging.info(f"Received data: {data.data}")
    response = requests.post("http://server:8001/predict", json={"data": data.features})
    prediction = response.json()
    return {"class": prediction["prediction"]}
