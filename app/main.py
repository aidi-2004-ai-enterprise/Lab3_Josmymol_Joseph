from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import xgboost as xgb
import pandas as pd
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penguin_api")

class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    male = "male"
    female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

app = FastAPI()

model = xgb.XGBClassifier()
model.load_model("app/data/model.json")

with open("app/data/meta.json") as f:
    meta = json.load(f)
    class_names = meta["classes"]
    expected_features = meta["features"]

@app.post("/predict")
def predict(features: PenguinFeatures):
    logger.info("Received prediction request: %s", features.dict())

    input_df = pd.DataFrame([features.dict()])
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in expected_features:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[expected_features]  # Ensure column order

    try:
        prediction = model.predict(input_df)[0]
        return {"prediction": class_names[prediction]}
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=400, detail="Prediction failed")
