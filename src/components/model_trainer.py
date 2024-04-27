import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LogisticRegression': LogisticRegression(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'SVC': SVC(),
                'SGDClassifier': SGDClassifier(),
                'GaussianNB': GaussianNB(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'ExtraTreesClassifier': ExtraTreesClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'XGBClassifier': XGBClassifier()
            }

            params = {
                'LogisticRegression': {'solver': ['liblinear']},
                'KNeighborsClassifier': {'n_neighbors': [100]},
                'SVC': {'gamma': ['auto'], 'kernel': ['linear'], 'probability': [True]},
                'SGDClassifier': {'loss': ['log_loss'], 'alpha': [0.1], 'random_state': [42]},
                'GaussianNB': {},
                'DecisionTreeClassifier': {},
                'RandomForestClassifier': {'max_depth': [10], 'random_state': [69]},
                'ExtraTreesClassifier': {'bootstrap': [False], 'criterion': ['entropy'], 'max_features': [1.0],
                                         'min_samples_leaf': [3], 'min_samples_split': [20], 'n_estimators': [100]},
                'GradientBoostingClassifier': {'n_estimators': [100], 'learning_rate': [1.0], 'max_depth': [6],
                                               'random_state': [69]},
                'XGBClassifier': {}
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            # get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found : " + best_model_name)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
