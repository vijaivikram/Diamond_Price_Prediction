import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.model_selection import train_test_split
import os
import sys
from pathlib import Path


class Data_Ingestion_Configuration:
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    train_data_path:str=os.path.join('artifacts','train_data.csv')
    test_data_path:str=os.path.join('artifacts','test_data.csv')
    
    
    
    
class Data_Ingestion:
    def __init__(self):
        self.ingestion_config = Data_Ingestion_Configuration()
        
    def initiate_data_ingestion(self):
        logging.info('Data ingestion process started')
        
        
        try:
            data=pd.read_csv(Path(os.path.join('notebooks/data','gemstone.csv')))
            logging.info('Data has been read')
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Raw data has been saved in artifacts')
            
            train_data,test_data = train_test_split(data,test_size=0.3)
            logging.info("Train test data has been splitted")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            logging.info('Train data has been saved in artifacts')
            
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info('Test data has been saved in artifacts')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
            
        except Exception as e:
            logging.info("Error occured during data ingestion stage")
            raise customexception(e,sys)