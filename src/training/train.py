import logging
import os
from datetime import datetime

import hydra
import joblib
import numpy as np
import pandas as pd
from names_generator import generate_name
from omegaconf import DictConfig
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate

from src.enums.enums import ModelType
from src.model.model import Model
from src.utils.dataframe.dataframe_utils import export_dataframe, read_dataframe

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(self, cfg, model):
        """
        Initializes the Trainer with a Model.
        """
        self.cfg = cfg
        self.model = model
        self.unique_id = generate_name()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._load_data()

    def train(self):
        """
        Train the model.
        """
        if self.cfg.trainer.feature_selection.enable_feature_selection:
            self._perform_feature_selection()

        cross_validation_metrics = self._cross_validate()
        self.model.fit(self.X_train, self.y_train)
        test_metrics = self.evaluate()
        self._save_metrics_to_csv(
            cv_metrics=cross_validation_metrics, test_metrics=test_metrics
        )
        self._save_model()

    def _perform_feature_selection(self):
        config = self.cfg.trainer.feature_selection
        if self.model.model_type == ModelType.XGBOOST:
            pass
        elif self.model.model_type == ModelType.RANDOM_FOREST:
            self.model.fit(self.X_train, self.y_train)
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            perm_importance = permutation_importance(
                estimator=self.model,
                X=self.X_test,
                y=self.y_test,
                scoring=scorer,
                n_repeats=config.permutation_importance.n_repeats,
                random_state=config.permutation_importance.random_state,
            )
            perm_importance_mean = perm_importance.importances_mean

            # Check if threshold is an integer
            if isinstance(config.permutation_importance.threshold, int):
                top_n_features = config.permutation_importance.threshold
                important_features_idx = np.argsort(perm_importance_mean)[
                    -top_n_features:
                ]
            else:
                threshold = config.permutation_importance.threshold * np.max(
                    perm_importance_mean
                )
                important_features_idx = np.where(
                    perm_importance_mean > threshold
                )[0]
            logging.info("Selected features -")
            logging.info(f"{self.X_train.columns[important_features_idx]}")
            self.X_train = self.X_train.iloc[:, important_features_idx]
            self.X_test = self.X_test.iloc[:, important_features_idx]

    def evaluate(self) -> dict:
        """
        Evaluate the model and return various metrics.
        """
        predictions = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions, squared=True)
        rmse = mean_squared_error(self.y_test, predictions, squared=False)
        r2 = r2_score(self.y_test, predictions)

        metrics = {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 2),
        }
        logging.info(f"Evaluation Metrics: {metrics}")

        return metrics

    def _cross_validate(self) -> dict:
        """
        Perform k-fold cross-validation and return evaluation metrics.
        """
        scoring = {
            "MAE": make_scorer(mean_absolute_error),
            "MSE": make_scorer(mean_squared_error, squared=True),
            "RMSE": make_scorer(mean_squared_error, squared=False),
            "R2": make_scorer(r2_score),
        }
        kf = KFold(
            n_splits=self.cfg.trainer.cross_validation.kfold.n_splits,
            shuffle=self.cfg.trainer.cross_validation.kfold.shuffle,
            random_state=self.cfg.trainer.cross_validation.kfold.random_state,
        )
        cv_results = cross_validate(
            estimator=self.model,
            X=self.X_train,
            y=self.y_train,
            scoring=scoring,
            cv=kf,
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
        train_datapath = self.cfg.trainer.data_path.train
        test_datapath = self.cfg.trainer.data_path.test
        target_column = self.cfg.trainer.target_column

        train_data = read_dataframe(train_datapath)
        test_data = read_dataframe(test_datapath)

        self.X_train = train_data.drop(columns=[target_column])
        self.y_train = train_data[target_column]
        self.X_test = test_data.drop(columns=[target_column])
        self.y_test = test_data[target_column]

    def _save_metrics_to_csv(self, cv_metrics: dict, test_metrics: dict):
        """
        Save cross-validation and test metrics to a CSV file.
        """
        filename = f"metrics_{self.timestamp}_{self.unique_id}.csv"
        filepath = os.path.join(self.cfg.trainer.metrics_folderpath, filename)

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

    def _save_model(self):
        """
        Save the trained model weights to a file.
        """
        identifier = f"{self.timestamp}_{self.unique_id}"
        models_folderpath = self.cfg.trainer.models_folderpath

        if not os.path.exists(models_folderpath):
            os.makedirs(models_folderpath)

        if self.model.model_type == ModelType.XGBOOST:
            filename = f"{self.cfg.model.model_name}_{identifier}.json"
            filepath = os.path.join(models_folderpath, filename)
            self.model.model.save_model(filepath)
        elif self.model.model_type == ModelType.RANDOM_FOREST:
            filename = f"{self.cfg.model.model_name}_{identifier}.pkl"
            filepath = os.path.join(models_folderpath, filename)
            joblib.dump(self.model, filepath)

        logging.info(f"Saved model - {filepath}")


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
    model_name = cfg.model.model_name
    model_config = getattr(cfg.model, model_name.lower())
    model = Model(cfg.model.model_name, **model_config)
    trainer = Trainer(cfg, model)
    trainer.train()

    return "Completed Model Training"


if __name__ == "__main__":
    run_standalone()
