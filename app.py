from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
import sys

if __name__ == "__main__":

    logging.info("The execution has started")

    try:
        obj = DataIngestion()
        obj.initiate_data_ingestion()
        logging.info("Data Ingestion is completed")
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)