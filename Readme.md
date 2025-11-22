# Car Price Prediction Project 🚗
## Table of Contents

1. Project Overview

2. Tech Stack

3. Architecture & Workflow

4. Data Ingestion

5. Data Transformation

6. Model Training & Evaluation

7. Prediction Pipeline

8. FastAPI Deployment

9. MLflow & DagsHub Integration

10. Dockerization

11. CI/CD Pipeline

12. Project Structure

13. References

## Project Overview

This project predicts car prices in Pakistan using historical car data. Features include:

- Company / Brand

- Car Model

- Year of Manufacture

- Kilometers Driven

- Fuel Type (Petrol / Diesel)

The project implements a complete ML workflow including data ingestion, preprocessing, modeling, deployment, and CI/CD.

## Tech Stack

- Backend & ML: Python, Pandas, NumPy, Scikit-learn, XGBoost

- API & Web: FastAPI, Jinja2 Templates, JavaScript, HTML/CSS

- Database: MySQL

- Experiment Tracking: MLflow, DagsHub

- Containerization: Docker

- CI/CD: GitHub Actions / GitLab CI

- Version Control: Git / DagsHub

## Architecture & Workflow
```bash
flowchart TD
    A[MySQL Database] --> B[Data Ingestion]
    B --> C[Data Transformation & Feature Engineering]
    C --> D[Model Training & Evaluation]
    D --> E[Prediction Pipeline]
    E --> F[FastAPI Backend]
    F --> G[Frontend UI]
    D --> H[MLflow & DagsHub Tracking]
    F --> I[Docker Container]
    I --> J[CI/CD Pipeline: GitHub Actions / GitLab CI]
```

## Data Ingestion

- Data is fetched from MySQL using Python (pymysql).

- Split data into train and test sets.

## Data Transformation

**Feature Engineering:**

- age = 2025 - year

- One-hot encode categorical features: company, name, fuel_type

- Scale numerical features: age, kms_driven

**Preprocessor object** is saved for later prediction: artifacts/preprocessor.pkl

## Model Training & Evaluation

Train a regression model (XGBoost / RandomForest) using the transformed dataset.

- Evaluate metrics:

- RMSE, MAE, R²

- Save trained model: artifacts/model.pkl

= Log metrics with MLflow.

## Prediction Pipeline

``predict_pipeline.py`` handles:

- Input validation using Pydantic

- Feature engineering (computing age)

- Transformation using saved preprocessor

- Prediction using saved model

Example:
```bash
pipeline = PredictionPipeline(model_path="artifacts/model.pkl",
                              preprocessor_path="artifacts/preprocessor.pkl")
prediction = pipeline.predict(input_df)

```

## FastAPI Deployment

**Endpoints:**

- / → Homepage with prediction form

- /predict → POST API for predictions

- /company → GET all companies

- /name/{company_name} → GET car models per company

- Frontend Integration:

- HTML/CSS/JS form

- AJAX calls to API endpoints

- Dynamic display of predicted price

Example request payload:

```bash
{
  "name": "Civic",
  "company": "Honda",
  "year": 2018,
  "kms_driven": 45000,
  "fuel_type": "Petrol"
}
```
## MLflow & DagsHub Integration

- Track experiments, metrics, and parameters with MLflow.

- Example:
```bash
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/<username>/<repo>.mlflow")
mlflow.log_param("model", "XGBRegressor")
mlflow.log_metric("RMSE", rmse)
mlflow.sklearn.log_model(best_model, "model")
```
**Benefits:**

- Model versioning

- Experiment comparison

- Collaboration via DagsHub

## Dockerization

Dockerfile:

```bash

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.mlproject.main:app", "--host", "0.0.0.0", "--port", "8000"]

```
**Commands**:
```bash
docker build -t car-price-predictor .
docker run -d -p 8000:8000 car-price-predictor
```

## CI/CD Pipeline

- Automate build, test, and deployment using GitHub Actions or GitLab CI.

**Sample GitHub Action workflow:**

```bash

name: CI/CD

on:
  push:
    branches: [main]

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
      - name: Build Docker
        run: docker build -t username/car-price-predictor .
      - name: Push Docker
        run: docker push username/car-price-predictor
```

### Project Structure

Car-Price-Predictor/
│
├─ src/
│   ├─ mlproject/
│   │   ├─ components/
│   │   │   ├─ data_ingestion.py
│   │   │   ├─ data_transformation.py
│   │   │   ├─ model_trainer.py
│   │   ├─ pipelines/
│   │   │   └─ prediction_pipeline.py
│   │   ├─ utils.py
│   │   ├─ logger.py
│   │   └─ main.py
│
├─ artifacts/
│   ├─ raw_data.csv
│   ├─ train.csv
│   ├─ test.csv
│   ├─ model.pkl
│   └─ preprocessor.pkl
│
├─ templates/
│   └─ index.html
├─ static/
│   └─ style.css
├─ requirements.txt
├─ Dockerfile
└─ README.md

