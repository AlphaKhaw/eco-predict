import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


"""
<class 'sklearn.linear_model._base.LinearRegression'>
<class 'sklearn.tree._classes.DecisionTreeRegressor'>
<class 'sklearn.ensemble._forest.RandomForestRegressor'>
<class 'xgboost.sklearn.XGBRegressor'>
<class 'lightgbm.sklearn.LGBMRegressor'>
<class 'interpret.glassbox._ebm._ebm.ExplainableBoostingRegressor'>
"""


def clean_column_names(dataframe):
    """
    Remove special JSON characters from column names.
    """
    # Define a regex pattern for JSON special characters
    pattern = r'[{,}"\[\]:\s]'

    # Replace characters in the column names matching the pattern
    dataframe.columns = [re.sub(pattern, "_", col) for col in dataframe.columns]

    return dataframe


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """
    Evaluate the model and return the metrics: MAE, MSE, RMSE, R-squared.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=True)
    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)

    results = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R-squared": r2}
    logging.info("Evaluation results:\n")
    for metric, score in results.items():
        print(f"{metric}: {score:.2f}")


def get_cross_val_scores(
    model,
    X_train,
    y_train,
    cross_validator,
    scoring_metrics: list = [
        "neg_mean_absolute_error",
        "neg_root_mean_squared_error",
        "r2",
    ],
):
    """
    Perform cross-validation using multiple metrics and log the results.
    """
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cross_validator,
        scoring=scoring_metrics,
        return_train_score=True,
    )

    logging.info("Cross Validation Results:")

    for metric_name, score_values in scores.items():
        if "train" in metric_name:
            mean_score = abs(np.mean(score_values))
            if "neg" in metric_name:
                print(f"Train {metric_name[6:]}: {mean_score:.2f} EUI")
            else:
                print(f"Train {metric_name[6:]}: {mean_score:.2f}")
        elif "test" in metric_name:
            mean_score = abs(np.mean(score_values))
            if "neg" in metric_name:
                print(f"Test {metric_name[5:]}: {mean_score:.2f} EUI")
            else:
                print(f"Test {metric_name[5:]}: {mean_score:.2f}")
