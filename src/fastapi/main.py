import logging
import os
import sys
from typing import Any, List

import pandas as pd
import uvicorn
import yaml
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from inference.inference import Inference

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def config() -> dict:
    """
    Load the configuration from a YAML file.

    Returns:
        dict: The configuration as a dictionary object.
    """
    with open("conf/base/inference.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
            return cfg
        except Exception as e:
            logging.info(
                f"An error occurred while loading the configuration: {e}"
            )
            raise e


# Initialize FastAPI
app = FastAPI()


class DataInput(BaseModel):
    """
    Pydantic BaseModel for validating input data.
    """

    percentageusageofled: float
    energyuseintensity_2017: float
    typeofairconditioningsystem_DistrictCoolingPlant: float
    averagemonthlybuildingoccupancyrate: float
    energyuseintensity_2018: float
    energyuseintensity_2019: float


class DataInputs(BaseModel):
    """
    Pydantic BaseModel for validating multiple input data.
    """

    data: List[DataInput]


@app.on_event("startup")
def startup_event() -> None:
    """
    Initializes the Inference instance upon application startup.
    """
    try:
        app.cfg = config()
        global MODEL
        MODEL = Inference(app.cfg)
        logging.info("Start server complete")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if MODEL is None:
        raise HTTPException(status_code=404, detail="Model weights not found")


@app.get("/")
async def health() -> dict:
    """
    Root API endpoint to check the health of the service.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"messages": "Hello from FastAPI!"}


@app.post("/predict_one/")
async def predict_one(data_input: DataInput) -> Any:
    if not data_input:
        raise HTTPException(
            status_code=400, detail="Input data cannot be empty."
        )
    try:
        input_data = [list(data_input.dict().values())]
        prediction = MODEL.predict(input_data)
        return {"prediction": prediction[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_many/")
async def predict_many(data_inputs: DataInputs) -> Any:
    if not data_inputs.data:
        raise HTTPException(
            status_code=400, detail="Input data cannot be empty."
        )
    try:
        input_data = [list(data.dict().values()) for data in data_inputs.data]
        predictions = MODEL.predict(input_data)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)) -> FileResponse:
    try:
        dataframe = pd.read_csv(file.file)
        selected_features = app.cfg["inference"]["selected_features"]
        if selected_features:
            dataframe = dataframe[selected_features]

        predictions = MODEL.predict(dataframe)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.fastapi.main:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="info",
    )
