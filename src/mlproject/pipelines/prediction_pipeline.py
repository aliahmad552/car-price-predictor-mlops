import sys
import pandas as pd
from pydantic import BaseModel, field_validator, Field
from typing import Literal
from src.mlproject.exception import CustomException
from src.mlproject.utils import load_object
from src.mlproject.logger import logging


# ---------------------------------------
# 1. Pydantic Model for Input Validation
# ---------------------------------------
class CarInput(BaseModel):
    name: str = Field(...,description="Car brand name must be one of the predefined brands")
    company: str = Field(...,description="Company name of the car")
    fuel_type: str = Field(...,description="Fuel type must be either 'Petrol' or 'Diesel'")
    year: int = Field(...,description="Manufacturing year of the car")
    kms_driven: int = Field(...,description="Total kilometers driven by the car")

    @field_validator("year")
    def validate_year(cls, v):
        if v < 1990 or v > 2025:
            raise ValueError("Year must be between 1990 and 2025")
        return v

    @field_validator("kms_driven")
    def validate_kms(cls, v):
        if v < 0:
            raise ValueError("kms_driven cannot be negative")
        return v

    # Clean & normalize company field
    @field_validator("company")
    def clean_company(cls, v):
        if not v.strip():
            raise ValueError("Company cannot be empty")
        return v.strip().title()   # Clean + Title Case
    

    # Convert into dataframe with engineered feature AGE
    def to_dataframe(self):
        age = 2025 - self.year
        return pd.DataFrame([{
            "name": self.name,
            "company": self.company,
            "fuel_type": self.fuel_type,
            "age": age,  # Final engineered feature
            "kms_driven": self.kms_driven
        }])


# ---------------------------------------
# 2. Prediction Pipeline
# ---------------------------------------
class PredictionPipeline:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading preprocessor and model")
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            logging.info("Transforming features using preprocessor")
            data_scaled = preprocessor.transform(features)

            logging.info("Predicting using the trained model")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            logging.error("Prediction error")
            raise CustomException(e, sys)
