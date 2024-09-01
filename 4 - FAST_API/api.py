import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc
import os
from fastapi.responses import JSONResponse


app = FastAPI(
    title="GetAround Project Fast Api",
    description="This is the API related to GetAround Project",
    version="0.1",
    contact={
        "name":"Yousra",
        "email":"yousraelkenfaoui@gmail.com",
    }

)

class PredictionFeatures(BaseModel):
    model_key: str  
    mileage: int
    engine_power: int 
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

@app.get("/")
async def index():
    message = "Hello world! This is my first API. If you want to learn more, check out documentation of the api at `/docs`"
    return message

@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Predictions of car rental prices per day
    """
    # Read data
    data = pd.DataFrame([{
        "model_key": predictionFeatures.model_key,
        "mileage": predictionFeatures.mileage,
        "engine_power": predictionFeatures.engine_power,
        "fuel": predictionFeatures.fuel,
        "paint_color": predictionFeatures.paint_color,
        "car_type": predictionFeatures.car_type,
        "private_parking_available": predictionFeatures.private_parking_available,
        "has_gps": predictionFeatures.has_gps,
        "has_air_conditioning": predictionFeatures.has_air_conditioning,
        "automatic_car": predictionFeatures.automatic_car,
        "has_getaround_connect": predictionFeatures.has_getaround_connect,
        "has_speed_regulator": predictionFeatures.has_speed_regulator,
        "winter_tires": predictionFeatures.winter_tires
    }])

    # Converst to float64 
    data["mileage"] = data["mileage"].astype(float)
    data["engine_power"] = data["engine_power"].astype(float)

    # Log model from mlflow
    logged_model = 'runs:/0396564035564ce19079b9a221e09f18/model'

    # Load model as a PyFuncModel
    try:
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print("Model loaded successfully")
        # Predictions
        prediction = loaded_model.predict(data)

        # Return prediction as JSON
        response = {"prediction": prediction.tolist()[0]}
    except mlflow.exceptions.MlflowException as e:
        response = {"error": str(e)}
        print(f"Erreur lors du chargement du modèle : {e}")   
         
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
