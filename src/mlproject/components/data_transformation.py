import sys
from dataclasses import dataclass
import os

from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object
from src.mlproject.logger import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            categorical_columns = ['name','company','fuel_type']
            numerical_columns = ['age','kms_driven']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ])
            return preprocessor
        


        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:

            train_path = pd.read_csv(train_path)
            test_path = pd.read_csv(test_path)
            

            print(train_path.shape[0])
            print(test_path.shape[0])
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'Price'

            # Feature engineering
            train_path["age"] = 2025 - train_path["year"]
            test_path["age"] = 2025 - test_path["year"]

            train_path.drop(columns=["year"], inplace=True)
            test_path.drop(columns=["year"], inplace=True)

            input_feature_train_df = train_path.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_path[target_column_name]

            input_feature_test_df = test_path.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_path[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)