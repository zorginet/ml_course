import os
import warnings

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(
    title="Bank Marketing Predictor",
    description="This API predicts whether a customer will subscribe to a term deposit based on their information.",
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = joblib.load(os.path.join(BASE_DIR, "production_pipe.joblib"))


class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "job": "technician",
                "marital": "married",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 300,
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": 1.1,
                "cons.price.idx": 93.994,
                "cons.conf.idx": -36.4,
                "euribor3m": 4.857,
                "nr.employed": 5191.0,
            }
        }
    }


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Bank Marketing Predictor API!",
        "description": "This API predicts whether a customer will subscribe to a term deposit based on their information.",
    }


@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_dict = data.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])

        input_df["was_contacted"] = input_df["pdays"].apply(
            lambda x: 1 if x != 999 else 0
        )

        cols_to_drop = ["duration", "emp.var.rate", "nr.employed", "pdays"]
        input_df = input_df.drop(columns=cols_to_drop, errors="ignore")

        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(prediction_proba),
            "status": "Target 1 (Yes)" if prediction == 1 else "Target 0 (No)",
        }
    except Exception as e:
        return {"error": str(e), "status": "Prediction failed"}
