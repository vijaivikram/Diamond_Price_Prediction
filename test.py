from src.DimondPricePrediction.pipelines.prediction_pipeline import CustomData

custom_data = CustomData(1.2,3.1,2.9,1.3,2.3,3.3,'ideal','F','VS2')

res = custom_data.get_data_as_dataframe()
print(res)