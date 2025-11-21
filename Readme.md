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

