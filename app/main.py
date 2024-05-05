import json
import joblib
import pandas as pd


def lambda_handler(event, context):
    # Extract input data from the request body    
    input = event['input']

    # Load the saved model from the pickle file
    loaded_lr_model = joblib.load('models/linear_regression.pkl')
    loaded_rf_model = joblib.load('models/random_forest.pkl')
    loaded_gb_model = joblib.load('models/gradient_boosting.pkl')

    # Using the loaded model for prediction
    X = pd.DataFrame([input], columns=["brand", "ram", "ssd", "hdd",
                     "no_of_cores", "no_of_threads", "cpu_brand", "gpu_brand", "screen_size", "screen_resolution", "os"])

    y_pred_lr = loaded_lr_model.predict(X)
    y_pred_rf = loaded_rf_model.predict(X)
    y_pred_gb = loaded_gb_model.predict(X)

    results = {
        "linear_regression": y_pred_lr.tolist(),
        "random_forest": y_pred_rf.tolist(),
        "gradient_boosting": y_pred_gb.tolist()
    }

    return {
        'statusCode': 200,
        'body': json.dumps({'results': json.dumps(results)})
    }
    
handler = lambda_handler

