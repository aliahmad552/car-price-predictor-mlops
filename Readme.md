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
