import joblib
import pandas as pd
from mangum import Mangum
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Item(BaseModel):
    input: list[int]


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from backend!"}


@app.post("/predict")
def read_item(request: Item):
    # Load the saved model from the pickle file
    loaded_lr_model = joblib.load('models/linear_regression.pkl')
    loaded_rf_model = joblib.load('models/random_forest.pkl')
    loaded_gb_model = joblib.load('models/gradient_boosting.pkl')

    # Using the loaded model for prediction
    X = pd.DataFrame([request.input], columns=["brand", "ram", "ssd", "hdd",
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
        "results": results
    }


handler = Mangum(app)
