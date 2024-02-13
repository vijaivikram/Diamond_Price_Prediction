from src.DimondPricePrediction.components.data_ingestion import Data_Ingestion
from src.DimondPricePrediction.components.data_transformation import Data_Transformation
from src.DimondPricePrediction.components.model_trainer import Train_model



import os
import sys

ingestion_obj = Data_Ingestion()
transformation_obj = Data_Transformation()
trainer_obj = Train_model()
train_data_path,test_data_path = ingestion_obj.initiate_data_ingestion()
train_arr,test_arr=transformation_obj.initiate_data_transformation(train_data_path,test_data_path)
trainer_obj.train_model(train_arr,test_arr)
