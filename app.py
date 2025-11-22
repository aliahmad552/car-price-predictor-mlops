from fastapi import FastAPI
from src.mlproject.pipelines.prediction_pipeline import PredictionPipeline, CarInput
from src.mlproject.logger import logging
import pandas as pd
import os
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

df = pd.read_csv("notebook/data/Cleaned_data.csv")
model_path = "artifacts/model.pkl"
preprocessor_path = "artifacts/preprocessor.pkl"


@app.get('/company')
def get_company():
    companies = df['company'].unique().tolist()
    return {'company': companies}

@app.get('/name/{company_name}')
def get_name(company_name: str):
    names = df[df['company'].str.lower() == company_name.lower()]['name'].unique().tolist()
    return {'company': company_name, 'name': names}

app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_car_price(car: CarInput):
    try:
        logging.info("Received prediction request")
        input_df = car.to_dataframe()
        logging.info(f"Input DataFrame: {input_df}")

        pipeline = PredictionPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
        prediction = pipeline.predict(input_df)

        logging.info(f"Prediction result: {prediction[0]}")
        return {"predicted_price": prediction[0]}
    except Exception as e:
        logging.error("Error during prediction", exc_info=True)
        return {"error": str(e)}