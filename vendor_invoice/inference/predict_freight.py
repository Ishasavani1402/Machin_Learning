import joblib
import pandas as pd

model_path = "models/predict_freight_model.pkl"

def load_model(model_path:str = model_path):
    # load the trained model
    with open(model_path , "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    # predict freight cost for new vendor

    model = load_model()
    input_df= pd.DataFrame(input_data)
    input_df['predicted_freight'] = model.predict(input_df).round()
    return input_df

if __name__ == '__main__':
    # example input data for prediction 
    sample_data ={
        # "Quantity": [10, 20, 15],
        "Dollars":[11200 , 4000,3000]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)