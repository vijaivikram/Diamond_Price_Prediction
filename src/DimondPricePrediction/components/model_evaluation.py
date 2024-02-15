import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.DimondPricePrediction.utils.utils import load_object



class Model_Evaluation:
    def __init__(self):
        pass



    def evaluate_metrics(self,y_test,y_pred):
        r2 = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))

        return r2,mae,rmse
    

    def initiate_model_evaluation(self,test_arr):
        try:
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            model_path = os.path.join('artifacts','model.pkl')
            model = load_object(model_path)

            mlflow.set_registry_uri("https://dagshub.com/vijaivikramiyyappan/Diamond_Price_Prediction.mlflow")

            url_type = urlparse(mlflow.get_tracking_uri()).scheme




            with mlflow.start_run():

                prediction = model.predict(x_test)

                r2,mae,rmse = self.evaluate_metrics(y_test,prediction)

                mlflow.log_metric('r2_score',r2)
                mlflow.log_metric('mae',mae)
                mlflow.log_metric('rmse',rmse)


                if url_type != 'file':

                    mlflow.sklearn.log_model(model,'model',registered_model_name='ml_model')

                else:

                    mlflow.sklearn.log_model(model,'model')




        except Exception as e:
            raise e
