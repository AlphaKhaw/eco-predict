import logging
import os
from datetime import datetime

import hydra
import pandas as pd
from names_generator import generate_name
from omegaconf import DictConfig
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate

from src.data_processing.feature_selector import FeatureSelector
from src.enums.enums import FeatureSelectionMethod
from src.model.model import Model
from src.utils.dataframe.dataframe_utils import export_dataframe
from src.utils.modelling.modelling import evaluate_model, load_data, save_model

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class Trainer:
    """
    Trainer class to perform model training.
    """

    def __init__(self, cfg: DictConfig, model: Model) -> None:
        """
        Initializes the Trainer with a Model.

        Args:
            cfg (DictConfig): Hydra configuration YAML.
            model (Model): Model class.
        """
        self.cfg = cfg.training.trainer
        self.model = model
        self.selector = FeatureSelector(cfg.data_processing.feature_selector)
        self.unique_id = generate_name()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.kf = KFold(
            n_splits=self.cfg.cross_validation.kfold.n_splits,
            shuffle=self.cfg.cross_validation.kfold.shuffle,
            random_state=self.cfg.cross_validation.kfold.random_state,
        )
        self.scoring = {
            "MAE": make_scorer(mean_absolute_error),
            "MSE": make_scorer(mean_squared_error, squared=True),
            "RMSE": make_scorer(mean_squared_error, squared=False),
            "R2": make_scorer(r2_score),
        }

        self._load_data()
        if self.cfg.enable_feature_selection:
            self._perform_feature_selection(
                config=cfg.data_processing.feature_selector
            )

    def train(self) -> None:
        """
        Train the model.
        """
        cross_validation_metrics = self._cross_validate()
        self.model.fit(self.X_train, self.y_train)
        test_metrics = self.evaluate()
        self._save_metrics_to_csv(
            cv_metrics=cross_validation_metrics, test_metrics=test_metrics
        )
        self._save_model()

    def _perform_feature_selection(self, config: DictConfig) -> None:
        """
        Feature Selection using FeatureSelector class.

        Args:
            config (DictConfig): Feature Selector configuration.
        """
        self.model.fit(self.X_train, self.y_train)
        if eval(config.method) == FeatureSelectionMethod.PERMUTATION_IMPORTANCE:
            important_features_idx = self.selector.select_features(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
                estimator=self.model,
            )
            logging.info("Selected features -")
            logging.info(f"{self.X_train.columns[important_features_idx]}")
            self.X_train = self.X_train.iloc[:, important_features_idx]
            self.X_test = self.X_test.iloc[:, important_features_idx]
        else:
            self.X_train = self.selector.select_features(
                X_train=self.X_train, y_train=self.y_train, estimator=self.model
            )
            self.X_test = self.selector.selector.transform(self.X_test)
            logging.info("Selected features -")
            logging.info(f"{self.selector.selector.get_feature_names_out()}")

    def evaluate(self) -> dict:
        """
        Evaluate the model and return the following metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R-squared (R^2)

        Returns:
            dict: Dictionary containing evaluation metrics
                  (MAE, MSE, RMSE, R-squared).
        """
        metrics = evaluate_model(
            model=self.model, X_test=self.X_test, y_test=self.y_test
        )

        return metrics

    def _cross_validate(self) -> dict:
        """
        Perform k-fold cross-validation and return evaluation metrics.
        """
        cv_results = cross_validate(
            estimator=self.model,
            X=self.X_train,
            y=self.y_train,
            scoring=self.scoring,
            cv=self.kf,
            return_train_score=False,
        )

        # Calculate mean and std of scores across folds
        cross_validation_metrics = {
            "MAE": (
                round(cv_results["test_MAE"].mean(), 2),
                round(cv_results["test_MAE"].std(), 2),
            ),
            "MSE": (
                round(cv_results["test_MSE"].mean(), 2),
                round(cv_results["test_MSE"].std(), 2),
            ),
            "RMSE": (
                round(cv_results["test_RMSE"].mean(), 2),
                round(cv_results["test_RMSE"].std(), 2),
            ),
            "R2": (
                round(cv_results["test_R2"].mean(), 2),
                round(cv_results["test_R2"].std(), 2),
            ),
        }
        logging.info(f"Cross-validation Metrics: {cross_validation_metrics}")

        return cross_validation_metrics

    def _load_data(self) -> None:
        """
        Load and preprocess data based on the provided Hydra configuration.
        """
        train_datapath = self.cfg.general.data_path.train
        test_datapath = self.cfg.general.data_path.test
        target_column = self.cfg.general.target_column

        self.X_train, self.y_train = load_data(
            filepath=train_datapath, target_column=target_column
        )
        self.X_test, self.y_test = load_data(
            filepath=test_datapath, target_column=target_column
        )

    def _save_metrics_to_csv(
        self, cv_metrics: dict, test_metrics: dict
    ) -> None:
        """
        Save cross-validation and test metrics to a CSV file.
        """
        filename = f"metrics_{self.timestamp}_{self.unique_id}.csv"
        filepath = os.path.join(self.cfg.general.metrics_folderpath, filename)

        metrics_dataframe = pd.DataFrame(
            {
                "Metric": ["MAE", "MSE", "RMSE", "R2"],
                "CV Mean": [
                    cv_metrics[metric][0]
                    for metric in ["MAE", "MSE", "RMSE", "R2"]
                ],
                "CV Std": [
                    cv_metrics[metric][1]
                    for metric in ["MAE", "MSE", "RMSE", "R2"]
                ],
                "Test": [
                    test_metrics[metric]
                    for metric in ["MAE", "MSE", "RMSE", "R2"]
                ],
            }
        )

        export_dataframe(dataframe=metrics_dataframe, output_filepath=filepath)
        logging.info(f"Saved metrics to {filepath}")

    def _save_model(self) -> None:
        """
        Save the trained model weights to a file.
        """
        identifier = f"{self.timestamp}_{self.unique_id}"
        models_folderpath = self.cfg.general.models_folderpath

        save_model(
            model=self.model,
            model_name=self.cfg.model_name,
            identifier=identifier,
            models_folderpath=models_folderpath,
        )


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone Trainer class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run Trainer class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str : Completion of Model Training.
    """
    model_name = cfg.training.trainer.model_name
    model_config = getattr(cfg.training.trainer, model_name.lower())
    model = Model(model_name, **model_config)
    trainer = Trainer(cfg, model)
    trainer.train()

    return "Completed Model Training"


if __name__ == "__main__":
    run_standalone()
