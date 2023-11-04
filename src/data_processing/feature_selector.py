import logging
from typing import Tuple, Union

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    SelectFromModel,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance

from src.enums.enums import FeatureSelectionMethod
from src.utils.dataframe.dataframe_utils import read_dataframe

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class FeatureSelector:
    """
    FeatureSelector class to perform feature selection.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a FeatureSelector object.

        Args:
            cfg (DictConfig): Hydra configuration YAML.
        """
        self.cfg = cfg
        self.param = cfg.param
        self.method = eval(cfg.method)
        if not isinstance(self.method, FeatureSelectionMethod):
            raise ValueError(f"Invalid method: {cfg.method}")

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        estimator: Union[object, None] = None,
    ) -> np.ndarray:
        """
        Select features from the input data using the specified method.

        Args:
            X_train (pd.DataFrame): Training data features.
            y_train (pd.Series): Training data labels.
            X_test (pd.DataFrame): Testing data features. Defaults to None.
            y_test (pd.Series): Testing data labels. Defaults to None.
            estimator (Union[object, None]): Base estimator for SelectFromModel.
                                             Defaults to None.

        Returns:
            np.ndarray: Transformed features based on the selected method.

        Raises:
            ValueError: Raise error when input method is not supported.
        """
        if self.method == FeatureSelectionMethod.PERCENTILE:
            self.selector = GenericUnivariateSelect(
                score_func=f_classif, mode="percentile", param=self.param
            )
        elif self.method == FeatureSelectionMethod.K_BEST:
            self.selector = GenericUnivariateSelect(
                score_func=f_classif, mode="k_best", param=self.param
            )
        elif self.method == FeatureSelectionMethod.FPR:
            self.selector = GenericUnivariateSelect(
                score_func=f_classif, mode="fpr", param=self.param
            )
        elif self.method == FeatureSelectionMethod.FDR:
            self.selector = GenericUnivariateSelect(
                score_func=f_classif, mode="fdr", param=self.param
            )
        elif self.method == FeatureSelectionMethod.MUTUAL_INFO:
            self.selector = GenericUnivariateSelect(
                score_func=mutual_info_classif, mode="k_best", param=self.param
            )
        elif self.method == FeatureSelectionMethod.FEATURE_IMPORTANCE:
            select_from_model_kwargs = self.cfg.select_from_model_kwargs
            self.selector = SelectFromModel(
                estimator=estimator, **select_from_model_kwargs
            )
        elif self.method == FeatureSelectionMethod.PERMUTATION_IMPORTANCE:
            perm_importance = permutation_importance(
                estimator=estimator,
                X=X_test,
                y=y_test,
                scoring=self.cfg.permutation_importance.scorer,
                n_repeats=self.cfg.permutation_importance.n_repeats,
                random_state=self.cfg.permutation_importance.random_state,
            )
            perm_importance_mean = perm_importance.importances_mean
            if isinstance(self.cfg.permutation_importance.threshold, int):
                top_n_features = self.cfg.permutation_importance.threshold
                important_features_idx = np.argsort(perm_importance_mean)[
                    -top_n_features:
                ]
            else:
                threshold = self.cfg.permutation_importance.threshold * np.max(
                    perm_importance_mean
                )
                important_features_idx = np.where(
                    perm_importance_mean > threshold
                )[0]
            return important_features_idx
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return self.selector.fit_transform(X_train, y_train)


def load_data(
    cfg: DictConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and preprocess data based on the provided Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Training and
                                                                 testing data
                                                                 and labels.
    """
    train_datapath = cfg.trainer.data_path.train
    test_datapath = cfg.trainer.data_path.test
    target_column = cfg.trainer.target_column

    train_data = read_dataframe(train_datapath)
    test_data = read_dataframe(test_datapath)

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone FeatureSelector class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None.
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    """
    Pass in Hydra configuration and run FeatureSelector class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]: Training and
                                                             testing data and
                                                             labels.
    """
    X_train, y_train, X_test, y_test = load_data(cfg)
    selector = FeatureSelector(cfg.data_processing.feature_selector)
    X_train_selected = selector.select_features(X_train, y_train)
    X_test_selected = selector.selector.transform(X_test)

    return X_train_selected, y_train, X_test_selected, y_test


if __name__ == "__main__":
    run_standalone()
