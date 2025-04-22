import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression,  Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info("Split training and tesing input data")
            Xtrain, Ytrain, Xtest, Ytest = (train_array[:, :-1],
                                             train_array[:,-1],
                                             test_array[:,:-1],
                                             test_array[:,-1])
            
            models = {
                "linear regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Catboost": CatBoostRegressor(verbose= False)
            }

            params = {
                "Decision tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boost":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear regression":{},
                "Ridge":{},
                "Lasso":{},
                "Catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNN":{
                    "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                    "n_neighbors": [1,2,3,4,5,6,7,8,9,10]
                }
            }
            

            model_report: dict = evaluate_models(Xtrain, Ytrain, Xtest, Ytest, models,params)

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best model not found")
            logging.info(f"Best found model on both training and testing dataset {best_model_name}")

            save_object(file_path= self.model_trainer_config.trained_model_file_path, obj= best_model)

            predicted = best_model.predict(Xtest)

            r2_scor = r2_score(Ytest, predicted)

            return r2_scor
        except Exception as e:
            raise CustomException(e, sys)