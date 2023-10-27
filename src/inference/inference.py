import logging
from enum import Enum

import joblib
import xgboost as xgb

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


def get_model_enum(model_type_str: str) -> ModelType:
    if isinstance(model_type_str, ModelType):
        return model_type_str

    return ModelType(model_type_str.lower())


class Inference:
    """
    Inference class to load model weights and perform inference.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Initialize a Inference object.

        Args:
            cfg (dict): Hydra configuration YAML.
        """
        self.cfg = cfg
        self.model_type = get_model_enum(cfg["inference"]["model"])
        self.model_weights_filepath = self.cfg["inference"][
            "model_weights_filepath"
        ]
        self._load_model_weights()

    def predict(self, data):
        """
        Make predictions with the selected model.
        """
        if self.model_type == ModelType.RANDOM_FOREST:
            return self.model.predict(data)

        elif self.model_type == ModelType.XGBOOST:
            return self.model.predict(xgb.DMatrix(data))

    def _load_model_weights(self) -> None:
        if self.model_type == ModelType.RANDOM_FOREST:
            self.model = joblib.load(self.model_weights_filepath)
        elif self.model_type == ModelType.XGBOOST:
            self.model = xgb.Booster()
            self.model.load_model(self.model_weights_filepath)
