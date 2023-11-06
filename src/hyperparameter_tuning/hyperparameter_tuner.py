import logging
import os
from datetime import datetime

import hydra
import optuna
import pandas as pd
from names_generator import generate_name
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error

from src.data_processing.feature_selector import FeatureSelector
from src.enums.enums import FeatureSelectionMethod
from src.model.model import Model
from src.utils.dataframe.dataframe_utils import export_dataframe
from src.utils.modelling.modelling import evaluate_model, load_data, save_model

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class HyperparameterTuner:
    """
    HyperparameterTuner class to perform model hyperparameter tuning.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the HyperparameterTuner with a Model.

        Args:
            cfg (DictConfig): Hydra configuration YAML.
            model (Model): Model class.
        """
        self.cfg = cfg.hyperparameter_tuning.hyperparameter_tuner
        self.model_name = self.cfg.general.model_name
        self.model_header = self.model_name.lower()
        self.direction = self.cfg.general.direction
        self.study_name = self.cfg.general.study_name
        self.n_trials = self.cfg.general.n_trials
        self.n_jobs = self.cfg.general.n_jobs
        self.search_space = self.cfg.search_space

        self.model = Model(
            cfg.training.trainer.model_name,
            **getattr(cfg.training.trainer, self.model_header),
        )
        self.sampler = optuna.samplers.TPESampler()
        self.pruner = optuna.pruners.MedianPruner()
        self.unique_id = generate_name()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._get_non_hyperparameters_params(cfg)
        self._load_data(cfg)
        if cfg.training.trainer.enable_feature_selection:
            self._perform_feature_selection(
                selector=FeatureSelector(cfg.data_processing.feature_selector),
                cfg=cfg.data_processing.feature_selector,
            )

    def objective(self, trial: optuna.trial) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.

        Args:
            trial (optuna.Trial): A trial object to sample hyperparameters.

        Returns:
            float: Root Mean Squared Error (RMSE) of the model on the validation
                   set.
        """
        n_estimators = trial.suggest_int(
            self.search_space[self.model_header].n_estimators.name,
            self.search_space[self.model_header].n_estimators.low,
            self.search_space[self.model_header].n_estimators.high,
        )
        max_depth = trial.suggest_int(
            self.search_space[self.model_header].max_depth.name,
            self.search_space[self.model_header].max_depth.low,
            self.search_space[self.model_header].max_depth.high,
        )
        min_samples_split = trial.suggest_float(
            self.search_space[self.model_header].min_samples_split.name,
            self.search_space[self.model_header].min_samples_split.low,
            self.search_space[self.model_header].min_samples_split.high,
        )
        min_samples_leaf = trial.suggest_float(
            self.search_space[self.model_header].min_samples_leaf.name,
            self.search_space[self.model_header].min_samples_leaf.low,
            self.search_space[self.model_header].min_samples_leaf.high,
        )
        max_features = trial.suggest_categorical(
            self.search_space[self.model_header].max_features.name,
            self.search_space[self.model_header].max_features.choices,
        )

        model = Model(
            model_type=self.model_name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **self.non_hyperparameter_params,
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        rmse = mean_squared_error(self.y_test, predictions, squared=False)

        return rmse

    def run_tuning(self):
        """
        Run hyperparameter tuning using Optuna and train the best model.

        This method performs hyperparameter tuning using the specified search
        space and trains a model with the best-found hyperparameters. It also
        evaluates the model's performance, saves the best model, and exports
        evaluation metrics to a CSV file.
        """
        study = optuna.create_study(
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name,
            direction=self.direction,
        )
        study.optimize(
            func=self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs
        )

        best_params = study.best_params
        best_params = {**best_params, **self.non_hyperparameter_params}

        self.model = Model(model_type=self.model_name, **best_params)
        self.model.fit(self.X_train, self.y_train)
        self._evaluate()
        self._save_model()
        self._save_metrics_to_csv()

    def _evaluate(self) -> None:
        """
        Evaluate the model and return the following metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R-squared (R^2)
        """
        self.metrics = evaluate_model(
            model=self.model, X_test=self.X_test, y_test=self.y_test
        )

    def _perform_feature_selection(
        self, selector: FeatureSelector, cfg: DictConfig
    ) -> None:
        """
        Feature Selection using FeatureSelector class.

        Args:
            selector: Instantiated FeatureSelector class.
            config (DictConfig): Feature Selector configuration.
        """
        self.model.fit(self.X_train, self.y_train)
        if eval(cfg.method) == FeatureSelectionMethod.PERMUTATION_IMPORTANCE:
            important_features_idx = selector.select_features(
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
            self.X_train = selector.select_features(
                X_train=self.X_train, y_train=self.y_train, estimator=self.model
            )
            self.X_test = selector.selector.transform(self.X_test)
            logging.info("Selected features -")
            logging.info(f"{selector.selector.get_feature_names_out()}")

    def _get_non_hyperparameters_params(self, cfg: DictConfig) -> None:
        """
        Get non-hyperparameter tuning parameters.

        Args:
            cfg (DictConfig): Hydra configuration YAML.
        """
        training_params = cfg.training.trainer[self.model_header]
        hyperparameter_params = self.cfg.search_space[self.model_header]
        self.non_hyperparameter_params = {
            key: value
            for key, value in training_params.items()
            if key not in hyperparameter_params
        }

    def _load_data(self, cfg: DictConfig) -> None:
        """
        Load and preprocess data based on the provided Hydra configuration.

        Args:
            cfg (DictConfig): Hydra configuration YAML.
        """
        train_datapath = cfg.training.trainer.general.data_path.train
        test_datapath = cfg.training.trainer.general.data_path.test
        target_column = cfg.training.trainer.general.target_column

        self.X_train, self.y_train = load_data(
            filepath=train_datapath, target_column=target_column
        )
        self.X_test, self.y_test = load_data(
            filepath=test_datapath, target_column=target_column
        )

    def _save_metrics_to_csv(self) -> None:
        """
        Save test metrics to a CSV file.
        """
        filename = f"metrics_{self.timestamp}_{self.unique_id}_tuned.csv"
        filepath = os.path.join(self.cfg.general.metrics_folderpath, filename)

        metrics_dataframe = pd.DataFrame(
            {
                "Test": [
                    self.metrics[metric]
                    for metric in ["MAE", "MSE", "RMSE", "R2"]
                ],
            }
        )

        export_dataframe(dataframe=metrics_dataframe, output_filepath=filepath)
        logging.info(f"Saved metrics to {filepath}")

    def _save_model(self) -> None:
        """
        Save the best hyperparameter-tuned model weights to a file.
        """
        identifier = f"{self.timestamp}_{self.unique_id}_tuned"
        models_folderpath = self.cfg.general.models_folderpath

        save_model(
            model=self.model,
            model_name=self.model_name,
            identifier=identifier,
            models_folderpath=models_folderpath,
        )


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone HyperparameterTuner class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run HyperparameterTuner class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str : Completion of Model Training.
    """
    tuner = HyperparameterTuner(cfg)
    tuner.run_tuning()

    return "Completed Hyperparameter Tuning"


if __name__ == "__main__":
    run_standalone()
