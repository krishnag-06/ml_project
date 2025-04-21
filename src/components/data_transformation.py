import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_column = ["reading_score", "writing_score"]
            categorical_column = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaling", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("impute", SimpleImputer(strategy = "most_frequent")),
                    ("encoding", OneHotEncoder()),
                    ("scaling", StandardScaler(with_mean= False))
                ]
            )

            logging.info(f"categorical columns: {categorical_column}")
            logging.info(f"numerical columns: {numerical_column}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_column),
                    ("cat_pipeline", categorical_pipeline, categorical_column)

                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("obtaining preproceesing object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column = "math_score"
            numerical_column = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(target_column, axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("applying preprocessing object on dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)