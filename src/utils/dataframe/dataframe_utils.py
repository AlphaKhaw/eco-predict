import logging
from pathlib import Path

import pandas as pd

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def read_dataframe(filepath: str) -> pd.DataFrame:
    """
    Read data from a CSV or Excel file and return it as a Pandas DataFrame.

    Args:
        filepath (str): The filepath of the file to be read. Supported formats
                        are CSV (.csv) and Excel (.xlsx).

    Returns:
        pd.DataFrame: The DataFrame containing the data from the file.

    Raises:
        TypeError: If the file format is not supported (only .csv and .xlsx are
                   supported).
    """
    if filepath.endswith("csv"):
        dataframe = read_from_csv(filepath)
    elif filepath.endswith("xlsx"):
        dataframe = read_from_excel(filepath)
    else:
        raise TypeError(
            "Invalid file type. Supported formats are .csv and .xlsx."
        )

    return dataframe


def read_from_csv(filepath: str) -> pd.DataFrame:
    """
    Read in CSV file without index and returns it as a Pandas DataFrame.
    Args:

        filepath (str): The filepath of the CSV file to be read.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    file_path = Path(filepath)
    if not Path(filepath).exists():
        raise FileNotFoundError(f"{filepath} not found.")

    if file_path.suffix.lower() != ".csv":
        raise TypeError("Invalid file type. Only .csv files are accepted.")

    dataframe = pd.read_csv(filepath)
    logging.info(f"Read CSV - {filepath}")

    return dataframe


def read_from_excel(filepath: str) -> pd.DataFrame:
    """
    Read in Excel file without index and return it as a Pandas DataFrame.

    Args:
        filepath (str): The filepath of the Excel file to be read.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the Excel file.
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"{filepath} not found.")

    if file_path.suffix.lower() not in (".xls", ".xlsx"):
        raise TypeError(
            "Invalid file type. Only .xls and .xlsx files are accepted."
        )

    dataframe = pd.read_excel(filepath)
    logging.info(f"Read Excel - {filepath}")

    return dataframe


def export_dataframe(dataframe: pd.DataFrame, output_filepath: str) -> None:
    """
    Export the DataFrame to a CSV or Parquet file without index in the
    specified folder path.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be exported as a CSV file.
        output_filepath (str): The file name of exported CSV file.

    Returns:
        None
    """
    try:
        filepath = Path(output_filepath)
        filepath.parents[0].mkdir(parents=True, exist_ok=True)

        if output_filepath.endswith(".csv"):
            if isinstance(dataframe, pd.core.frame.DataFrame):
                dataframe.to_csv(output_filepath, index=False)
            else:
                csv_kwargs = {
                    "filename": filepath,
                    "single_file": True,
                    "index": False,
                    "escapechar": "\\",
                }
                dataframe.to_csv(**csv_kwargs)
            logging.info(f"Exported CSV file: {filepath}")

        elif output_filepath.endswith(".parquet"):
            parquet_kwargs = {"path": filepath, "write_metadata_file": True}
            dataframe.to_parquet(**parquet_kwargs)
            logging.info(f"Exported Parquet file: {filepath}")

        else:
            raise ValueError("Output format currently not supported")

    except Exception as error:
        logging.error(f"An error occurred while exporting: {error}")
        raise
