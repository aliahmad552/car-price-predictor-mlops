from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
import sys

if __name__ == "__main__":

    logging.info("The execution has started")

    try:
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        logging.info("Data Ingestion is completed")

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data Transformation is completed")

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model Training is completed")


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)