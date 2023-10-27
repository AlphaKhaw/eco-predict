import logging
import os
import sys

import hydra
import joblib
import xgboost as xgb
from omegaconf import DictConfig

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from enums.enums import ModelType

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class Inference:
    """
    Inference class to load model weights and perform inference.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a Inference object.

        Args:
            cfg (DictConfig): Hydra configuration YAML.
        """
        self.cfg = cfg
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
        model_name = self.cfg.inference.model
        model_weights_filepath = self.cfg.inference.model_weights_filepath

        if model_name == ModelType.RANDOM_FOREST:
            self.model = joblib.load(model_weights_filepath)
        elif model_name == ModelType.XGBOOST:
            self.model = xgb.Booster()
            self.model.load_model(model_weights_filepath)


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone Inference class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> Inference:
    """
    Pass in Hydra configuration and run Inference class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        Inference: An instance of the Inference class.
    """
    return Inference(cfg)


if __name__ == "__main__":
    run_standalone()
