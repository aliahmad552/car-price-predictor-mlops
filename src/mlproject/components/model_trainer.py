import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_model
import os
import mlflow
from urllib.parse import urlparse
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Support Vector Regressor": SVR(),
                "XGBoost": XGBRegressor()
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(best_model_score)
            print(best_model_name)

            # Authentication
            os.environ["MLFLOW_TRACKING_USERNAME"] = "aliahmad552"
            os.environ["MLFLOW_TRACKING_PASSWORD"] = "738cd055a1ed8cfe2b2ff2eaddea249c2471b62a"

            # Correct DagsHub tracking URI
            mlflow.set_tracking_uri("https://dagshub.com/aliahmad552/car-price-predictor-mlops.mlflow")

            # Start the MLflow experiment
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)

                # DagsHub DOES NOT support model registry
                mlflow.sklearn.log_model(best_model,"model")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
