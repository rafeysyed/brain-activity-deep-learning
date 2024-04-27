import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTEENN

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

from collections import Counter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation

        """

        try:

            # Define numerical feature columns
            numerical_columns = []
            for i in range(1, 179):
                numerical_columns.append("X" + str(i))

            # Define the pipeline for data transformation
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns)
                ]
            )

            # Combine resampling and cleaning methods using SMOTEENN
            resampler = SMOTEENN()

            # Create the pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('resampler', resampler)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, data_path):
        """
        This function is responsible for data transformation.

        """
        try:
            df = pd.read_csv(data_path)

            # Remove the 'Unnamed' column
            df.drop(columns=['Unnamed'], inplace=True)

            # One-Hot Encoding the target variable
            df['y'] = df['y'].replace([2, 3, 4, 5], 0)

            # Separate input features (X) and target column (y)
            X = df.drop(columns=['y'])
            y = df['y']

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info(
                f"Applying preprocessing object on the dataframe."
            )

            # Fit the data transformer
            preprocessing_obj.fit(X)

            # Fit and transform the data
            X_transformed = preprocessing_obj.fit_transform(X)

            # Convert DataFrame to 2D numpy array
            X_array = np.array(X_transformed)
            y_array = np.array(y).reshape(-1, 1)  # Reshape to column vector

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.4, random_state=42)

            # Concatenate X_train and y_train along the second axis to create train_arr
            train_arr = np.concatenate((X_train, y_train), axis=1)

            # Concatenate X_test and y_test along the second axis to create test_arr
            test_arr = np.concatenate((X_test, y_test), axis=1)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    """
----------------------------------------------------------------------------------------------------
    """

    # def get_data_transformer_object(self):
    #     """
    #     This function si responsible for data transformation
    #
    #     """
    #     cols = []
    #     for i in range(1, 179):
    #         cols.append("X"+str(i))
    #
    #     try:
    #
    #         num_pipeline = Pipeline(
    #             steps=[
    #                 # ("imputer", SimpleImputer(strategy="median")),
    #                 ("smenn", SMOTEENN()),
    #                 ("scaler", StandardScaler())
    #
    #             ]
    #         )
    #
    #         # cat_pipeline = Pipeline(
    #         #
    #         #     steps=[
    #         #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #         #         ("one_hot_encoder", OneHotEncoder()),
    #         #         ("scaler", StandardScaler(with_mean=False))
    #         #     ]
    #         #
    #         # )
    #
    #         # logging.info(f"Categorical columns: {categorical_columns}")
    #         # logging.info(f"Numerical columns: {numerical_columns}")
    #
    #         preprocessor = ColumnTransformer(
    #             [
    #                 ("num_pipeline", num_pipeline, cols),
    #                 # ("cat_pipelines", cat_pipeline, categorical_columns)
    #
    #             ]
    #
    #         )
    #
    #         return preprocessor
    #
    #     except Exception as e:
    #         raise CustomException(e, sys)

    # def initiate_data_transformation(self, train_path, test_path):
    #     """
    #     This function is responsible for data transformation.
    #
    #     """
    #     try:
    #         df = pd.read_csv
    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)
    #
    #         logging.info("Read train and test data completed")
    #
    #         train_df['y'] = train_df['y'].replace([2, 3, 4, 5], 0)
    #         test_df['y'] = test_df['y'].replace([2, 3, 4, 5], 0)
    #
    #         logging.info("Obtaining preprocessing object")
    #
    #         preprocessing_obj = self.get_data_transformer_object()
    #
    #         # column_names = train_df.columns.tolist()
    #         # column_names.remove('Unnamed')
    #         # column_names.remove('y')
    #
    #         target_column_name = "y"
    #         # numerical_columns = column_names
    #
    #         input_feature_train_df = train_df.drop(columns=[target_column_name, 'Unnamed'], axis=1)
    #         target_feature_train_df = train_df[target_column_name]
    #
    #         input_feature_test_df = test_df.drop(columns=[target_column_name, 'Unnamed'], axis=1)
    #         target_feature_test_df = test_df[target_column_name]
    #
    #         logging.info(
    #             f"Applying preprocessing object on training dataframe and testing dataframe."
    #         )
    #
    #         input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
    #         input_feature_train_arr = preprocessing_obj.fit_resample(input_feature_train_df, target_feature_train_df)
    #         input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
    #
    #         train_arr = np.c_[
    #             input_feature_train_arr, np.array(target_feature_train_df)
    #         ]
    #         test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
    #
    #         logging.info(f"Saved preprocessing object.")
    #
    #         save_object(
    #
    #             file_path=self.data_transformation_config.preprocessor_obj_file_path,
    #             obj=preprocessing_obj
    #
    #         )
    #
    #         return (
    #             train_arr,
    #             test_arr,
    #             self.data_transformation_config.preprocessor_obj_file_path
    #         )
    #     except Exception as e:
    #         raise CustomException(e, sys)
