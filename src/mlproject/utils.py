import os
import pandas as pd
from src.mlproject.exception import CustomException
import sys
import dill
from src.mlproject.logger import logging
from dotenv import load_dotenv
from sklearn.metrics import r2_score

import pymysql
import pickle

load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')

def read_sql_data():
    """
    Reads data from a SQL database and returns it as a pandas DataFrame.

    """
    logging.info("Attempting to connect to MySQL database")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db
        )

        logging.info(f"Connection Established: {mydb}")

        df = pd.read_sql_query("SELECT * FROM car_data",mydb)
        print(df.head())

        return df

    except Exception as ex:
        logging.error("Error while connecting to MySQL", exc_info=True)
        raise CustomException(ex,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)