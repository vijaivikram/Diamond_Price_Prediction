from src.DimondPricePrediction.components.data_ingestion import Data_Ingestion

import os
import sys

ingestion_obj = Data_Ingestion()
ingestion_obj.initiate_data_ingestion()