import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass


from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.utils.utils import save_object 


from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class Data_Transformation_Configuration:
    preprocessor_object_path:str=os.path.join('artifacts','preprocessor.pkl')
    
    
    
class Data_Transformation:
    def __init__(self):
        self.transformation_configuration = Data_Transformation_Configuration()
    
    
    def data_transformation(self):
        
        try:
            logging.info('Data transformation building started')
            
            
            categorical_columns=['cut', 'color', 'clarity']
            numerical_columns=['carat', 'depth', 'table', 'x', 'y', 'z']
            
            
            
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['J','I','H','G','F','E','D']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
    
            
            
            numerical_pipeline = Pipeline(
                steps=[
                    ('simple_imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            
            
            categorical_pipeline = Pipeline(
                steps=[
                    ('simple_imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )
            
            
            
            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('categorical_pipeline',categorical_pipeline,categorical_columns)       
            ])
            
            
            return preprocessor
        
        
        
        except Exception as e:
            logging.info('Error occured during data transformation building')
            raise customexception(e,sys)
            
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            logging.info('train and test has been read')
            
            preprocessing_object = self.data_transformation()
            
            target_feature = 'price'
            
            drop_columns = [target_feature,'id']
            
            input_feature_train_data = train_data.drop(columns=drop_columns,axis=1)
            target_feature_train_data = train_data[target_feature]
            
            input_feature_test_data = test_data.drop(columns=drop_columns,axis=1)
            target_feature_test_data = test_data[target_feature]
            
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_data) 
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_data)
            
            logging.info('Applied preprocessing techniques on X-train and X-test data')
            
            train_arr =np.c_[input_feature_train_arr,np.array(target_feature_train_data)] 
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            
            save_object(file_path=self.transformation_configuration.preprocessor_object_path,obj=preprocessing_object)
            
            logging.info('Preprocessing object saved in artifacts')
            
            
            
            return (
                train_arr,
                test_arr
            )
            
            
        except Exception as e:
            logging.info("Error occured during data transformation stage")
            raise customexception(e,sys)
            
            
            
    