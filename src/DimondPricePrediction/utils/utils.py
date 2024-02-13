import pandas as pd
import numpy as np
import os
import sys
import pickle
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info('Error occured in save object in utils')
        raise customexception(e,sys)
    


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error occured during load object in utils')
        raise customexception(e,sys)



def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            
            y_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test,y_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
        
        
    except Exception as e:
        logging.info('Error occured in evaluate model in utils')
        raise customexception(e,sys)