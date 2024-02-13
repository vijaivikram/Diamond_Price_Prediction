import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass



from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.utils.utils import save_object,evaluate_model


from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet



@dataclass
class Model_Training_Configuration:
    model_path:str=os.path.join('artifacts','model.pkl')
    
    
    
    
class Train_model:
    def __init__(self):
        self.model_configuration = Model_Training_Configuration()
        
        
    def train_model(self,train_arr,test_arr):
        logging.info('Model training started')
        
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info('Splitted dependant and independant variables from train and test data')
            
            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
            }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            
            logging.info('Model evaluated')
            logging.info(f'Model report : {model_report}' )
            
            print(model_report)
            
            print("="*50)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            print(best_model_name)
            best_model = models[best_model_name]
            
            print(f"Best Model Found : {best_model_name}, Model Score : {best_model_score}")
            
            logging.info(f"Best Model Found : {best_model_name}, Model Score : {best_model_score}")
            
            save_object(file_path=self.model_configuration.model_path,obj=best_model)
            
            
            
        except Exception as e:
            logging.info("Error occured while training the model")
            raise customexception(e,sys)