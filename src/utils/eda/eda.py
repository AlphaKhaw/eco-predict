import logging
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, shapiro, zscore
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def drop_columns(
    dataframe: pd.DataFrame, columns: Union[list, str]
) -> pd.DataFrame:
    """
    Drop specified columns from a dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe from which columns are
                                  to be dropped.
        columns (Union[list, str]): Column or list of columns to be dropped
                                    from the dataframe.

    Returns:
        pd.DataFrame: DataFrame after dropping the specified columns.

    Raises:
        TypeError: If 'columns' is neither a string nor a list.
    """
    if isinstance(columns, str):
        dataframe.drop(columns=[columns], inplace=True)
        logging.info(f"Dropped column - {columns}")
    elif isinstance(columns, list):
        dataframe.drop(columns=columns, inplace=True)
        logging.info(f"Dropped columns - {columns}")
    else:
        raise TypeError("'columns' input should be string or list datatype.")

    return dataframe


def find_best_imputation_for_feature(
    dataframe: pd.DataFrame, column: str
) -> dict:
    # Define imputers
    imputation_methods = {
        "median": SimpleImputer(strategy="median"),
        "mean": SimpleImputer(strategy="mean"),
        "knn": KNNImputer(n_neighbors=5),
        "iterative": IterativeImputer(max_iter=10, random_state=42),
    }

    results = {}
    original_data = dataframe[column].dropna()

    # Iterate over each imputation method
    for method_name, imputer in imputation_methods.items():
        # Copy the original dataframe to ensure no modifications in-place
        data_copy = dataframe[[column]].copy()
        imputed_data = imputer.fit_transform(data_copy).ravel()

        # Using KS test to compare the original (non-NaN) values with
        # imputed values
        D, p_value = ks_2samp(original_data, imputed_data)
        results[method_name] = {"D": D, "p_value": p_value}

    # Find the method with the highest p-value and the smallest D value
    best_method = max(
        results, key=lambda x: (results[x]["p_value"], -results[x]["D"])
    )
    best_method_info = {"method": best_method, **results[best_method]}
    logging.info(
        f"""Kolmogorov-Smirnov test results for the feature `{column}`:
        Method: {best_method_info['method']}
        D: {best_method_info['D']}
        p-value: {best_method_info['p_value']}
        """
    )

    return best_method_info


def rename_columns(dataframe: pd.DataFrame, mapper: dict) -> pd.DataFrame:
    """
    Rename columns of a DataFrame based on a provided mapping.

    Args:
        dataframe (pd.DataFrame): The input DataFrame whose columns are to be
                                  renamed.
        mapper (dict): A dictionary mapping from current column names to new
                       column names.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    dataframe.rename(columns=mapper, inplace=True)

    return dataframe


def identify_iqr_outliers(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """
    Identify outliers in a DataFrame's column using the IQR method.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column name in which to search for outliers.

    Returns:
        pd.Series: A boolean Series indicating True for outliers and False
                   otherwise.
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers_indices = np.where(
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    )[0]
    outliers_values = dataframe[column].iloc[outliers_indices]

    logging.info(
        f"""For feature '{column}', IQR detected outliers values are:
        {list(outliers_values)}
        """
    )

    return outliers_values


def identify_z_score_outliers(dataframe, column, threshold=3):
    """Identify outliers in a DataFrame's column using the Z-score method.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column name in which to search for outliers.
    - threshold (float): The Z-score threshold to use (default is 3).

    Returns:
        pd.Series: A boolean Series indicating True for outliers and False
                   otherwise.
    """
    z_scores = zscore(dataframe[column])

    outliers_indices = np.where(np.abs(z_scores) > threshold)
    outliers_values = dataframe[column].iloc[outliers_indices]

    logging.info(
        f"""For feature '{column}', Z-score detected outliers values are:
        {list(outliers_values)}
        """
    )

    return outliers_values


def impute_with_mean(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    dataframe[column] = imputer.fit_transform(dataframe[[column]])

    return dataframe


def impute_with_median(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    dataframe[column] = imputer.fit_transform(dataframe[[column]])

    return dataframe


def impute_with_iterative(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    imputer = IterativeImputer(max_iter=10, random_state=42)
    dataframe[column] = imputer.fit_transform(dataframe[[column]])

    return dataframe


def impute_with_knn(
    dataframe: pd.DataFrame, column: str, n_neighbors: int = 5
) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    dataframe[column] = imputer.fit_transform(dataframe[[column]])

    return dataframe


def impute_missing_values(
    dataframe: pd.DataFrame, column: str, new_value: Union[float, str]
) -> pd.DataFrame:
    """
    Impute missing values in a specific column of a dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe with missing values.
        column (str): The column in which missing values need to be imputed.
        new_value (Union[float, str]): The value to be used for imputation.

    Returns:
        pd.DataFrame: DataFrame after imputation.
    """
    dataframe[column] = dataframe[column].fillna(new_value)
    logging.info(f"Impute missing values for {column}")

    return dataframe


def test_normality(dataframe: pd.DataFrame, column: str) -> float:
    _, p_value = shapiro(dataframe[column].dropna())
    if p_value > 0.05:
        logging.info(f"The feature, `{column}`, is normally distributed")
    else:
        logging.info(f"The feature, `{column}`, is not normally distributed")

    return p_value


def test_ks_similarity(original_data, imputed_data):
    _, p_value = ks_2samp(original_data.dropna(), imputed_data)
    if p_value > 0.05:
        logging.info("Two samples are from the same distribution")
    else:
        logging.info("Two samples are from different distributions")

    return p_value
