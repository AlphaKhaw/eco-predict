import logging
import os
from typing import Union

import hydra
import pandas as pd
from omegaconf import DictConfig, ListConfig
from sklearn.preprocessing import OneHotEncoder

from src.enums.enums import IMPUTATION_FUNCTIONS, ImputationStrategy
from src.utils.dataframe.dataframe_utils import export_dataframe, read_dataframe

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataPreprocessor:
    """
    DataPreprocessor class to perform data cleaning.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataPreprocessor object.

        Args:
            cfg (DictConfig): Hydra configuration YAML.

        Returns:
            None
        """
        self.cfg = cfg

    def merge_datasets(
        self, output_folderpath: str, suffix: str, common_identifier: str
    ) -> None:
        """
        Merge multiple datasets in the output directory. Iterates through all
        CSV files in the specified directory, merges them  on a common
        identifier, and exports the merged data.

        If there's only one dataset, it renames it. Drops the common identifier
        column after merging.

        Args:
            output_folderpath (str): Path to the folder containing the
                                     individual preprocessed datasets.
            suffix (str): Output suffix of preprocessed file.
            common_identifier (str): Common column to use for merging.
        """
        # List all .csv files in the output directory
        files = [
            f
            for f in os.listdir(output_folderpath)
            if f.endswith(".csv")
            and os.path.isfile(os.path.join(output_folderpath, f))
            and f != suffix
        ]

        # No files to merge
        if not files:
            return

        # If there's only one dataframe, rename it to processed.csv
        if len(files) == 1:
            os.rename(
                os.path.join(output_folderpath, files[0]),
                os.path.join(output_folderpath, suffix),
            )
            filepath = os.path.join(output_folderpath, suffix)
            dataframe = read_dataframe(filepath=filepath)
            dataframe = self._drop_columns(
                dataframe=dataframe, columns=[common_identifier]
            )
            export_dataframe(dataframe=dataframe, output_filepath=filepath)
            return

        # Read each file and append to a list
        dataframes = []
        for file in files:
            filepath = os.path.join(output_folderpath, file)
            dataframe = read_dataframe(filepath=filepath)
            dataframes.append((file, dataframe))
            os.remove(filepath)

        # If 2020 dataset exists, use pd.merge on 'buildingaddress',
        # otherwise use pd.concat
        if "2020_preprocessed.csv" in files:
            base_file, base_dataframe = next(
                (f, dataframe)
                for f, dataframe in dataframes
                if f == "2020_preprocessed.csv"
            )
            for file, dataframe in dataframes:
                if file != base_file:
                    base_dataframe = pd.merge(
                        base_dataframe,
                        dataframe,
                        on=common_identifier,
                        how="inner",
                    )
            combined_dataframe = base_dataframe
        else:
            combined_dataframe = pd.concat(
                [dataframe for _, dataframe in dataframes], ignore_index=True
            )

        # Drop 'buildingaddress' column after merging
        if common_identifier in combined_dataframe.columns:
            combined_dataframe.drop(columns=[common_identifier], inplace=True)

        export_dataframe(
            dataframe=combined_dataframe,
            output_filepath=os.path.join(output_folderpath, "processed.csv"),
        )

    def preprocess_data(self, dataset: Union[int, str]) -> None:
        """
        Preprocess the input dataset based on configurations:
        - Reads in configurations
        - Drop columns
        - Removing unwanted symbols
        - Converting dtypes
        - Imputation of categorical and numerical features
        - Encoding of binary, nominal and ordinal columns
        - Renames columns
        - Exports the processed dataframe with a specified suffix.

        Args:
            dataset (Union[int, str]): The dataset identifier or name.
        """
        # Read in configurations
        output_folderpath = self.cfg.general.output_folderpath
        output_suffix = self.cfg.general.suffix
        input_filepath = self.cfg.datasets[dataset].input_filepath
        preprocessing_steps = self.cfg.datasets[dataset].preprocessing_steps
        columns_to_drop = preprocessing_steps.drop_columns
        remove_symbols_dict = preprocessing_steps.remove_symbols
        convert_dtypes_dict = preprocessing_steps.convert_dtypes
        impute_categorical_dict = preprocessing_steps.impute_categorical_columns
        impute_numerical_dict = preprocessing_steps.impute_numerical_columns
        encode_dict = preprocessing_steps.encode_columns
        column_name_mapping = preprocessing_steps.rename_columns

        # Read in dataframe
        dataframe = read_dataframe(filepath=input_filepath)

        # Drop columns
        dataframe = self._drop_columns(
            dataframe=dataframe, columns=columns_to_drop
        )

        # Remove symbols
        dataframe = self._remove_symbols(
            dataframe=dataframe, remove_symbols_dict=remove_symbols_dict
        )

        # Convert dtypes
        dataframe = self._convert_dtypes(
            dataframe=dataframe, convert_dtypes_dict=convert_dtypes_dict
        )

        # Impute categorical columns
        dataframe = self._impute_categorical_columns(
            dataframe=dataframe, impute_categorical_dict=impute_categorical_dict
        )

        # Impute numerical columns
        dataframe = self._impute_numerical_columns(
            dataframe=dataframe, impute_numerical_dict=impute_numerical_dict
        )

        # Encode binary and ordinal columns
        dataframe = self._encode_binary_ordinal_columns(
            dataframe=dataframe, config=encode_dict.binary
        )
        dataframe = self._encode_binary_ordinal_columns(
            dataframe=dataframe, config=encode_dict.ordinal
        )

        # Encode nominal columns
        dataframe = self._encode_nominal_columns(
            dataframe=dataframe, nominal_config=encode_dict.nominal
        )

        # Rename columns
        dataframe.rename(columns=column_name_mapping, inplace=True)

        # Export dataframe
        dataset_suffix = f"{dataset}_{output_suffix}"
        output_filepath = os.path.join(output_folderpath, dataset_suffix)
        export_dataframe(dataframe=dataframe, output_filepath=output_filepath)

    def _encode_nominal_columns(
        self, dataframe: pd.DataFrame, nominal_config: ListConfig
    ) -> pd.DataFrame:
        """
        Encode nominal columns using one-hot encoding.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            nominal_config (ListConfig): List of columns to encode.

        Returns:
            pd.DataFrame: DataFrame with nominal columns encoded.
        """
        encoder = OneHotEncoder(drop="first")
        for column in nominal_config:
            encoded = encoder.fit_transform(dataframe[[column]]).toarray()
            dataframe = pd.concat(
                [
                    dataframe,
                    pd.DataFrame(
                        encoded, columns=encoder.get_feature_names_out([column])
                    ),
                ],
                axis=1,
            )
            dataframe = self._drop_columns(
                dataframe=dataframe, columns=[column]
            )

        return dataframe

    def _impute_categorical_columns(
        self, dataframe: pd.DataFrame, impute_categorical_dict: DictConfig
    ) -> pd.DataFrame:
        """
        Impute missing values in categorical columns.

        Args:
            dataframe (pd.DataFrame): Input dataframe.
            impute_categorical_dict (DictConfig): Dictionary of columns and
                                                  their imputation config.

        Returns:
            pd.DataFrame: DataFrame with imputed values.
        """
        for column, impute_config in impute_categorical_dict.items():
            dataframe[column].fillna(impute_config.value, inplace=True)

            if column == "buildingsize":
                dataframe = self._process_buildingsize(
                    dataframe=dataframe, column=column, config=impute_config
                )
            elif column == "greenmarkversion":
                dataframe = self._process_greenmarkversion(
                    dataframe=dataframe, column=column, config=impute_config
                )

            logging.info(f"Impute categorical column - {column}")

        return dataframe

    @staticmethod
    def _convert_dtypes(
        dataframe: pd.DataFrame, convert_dtypes_dict: DictConfig
    ) -> pd.DataFrame:
        for column, dtype in convert_dtypes_dict.items():
            if dataframe[column].dtype != dtype:
                dataframe[column] = dataframe[column].astype(dtype)
                logging.info(f"Converted dtype for column - {column}")

        return dataframe

    @staticmethod
    def _drop_columns(
        dataframe: pd.DataFrame, columns: ListConfig
    ) -> pd.DataFrame:
        """
        Drop specified columns from a dataframe.

        Args:
            dataframe (pd.DataFrame): The input dataframe from which columns
                                      are to be dropped.
            columns (ListConfig): List of columns to be dropped from the
                                  dataframe.

        Returns:
            pd.DataFrame: DataFrame after dropping the specified columns.
        """
        dataframe.drop(columns=columns, inplace=True)
        logging.info(f"Dropped columns - {columns}")

        return dataframe

    @staticmethod
    def _encode_binary_ordinal_columns(
        dataframe: pd.DataFrame, config: DictConfig
    ) -> pd.DataFrame:
        """
        Encodes binary ordinal columns in the given dataframe based on the
        provided configuration.

        Args:
            dataframe (pd.DataFrame): The input dataframe to encode.
            config (DictConfig): Configuration mapping for columns to ordinal
                                 values.

        Returns:
            pd.DataFrame: The dataframe with encoded binary ordinal columns.
        """
        for column, map_dict in config.items():
            dataframe[column] = dataframe[column].map(map_dict)

        return dataframe

    @staticmethod
    def _impute_numerical_columns(
        dataframe: pd.DataFrame, impute_numerical_dict: DictConfig
    ) -> pd.DataFrame:
        """
        Imputes numerical columns in the given dataframe based on the provided
        imputation configuration.

        Args:
            dataframe (pd.DataFrame): The input dataframe to impute.
            impute_numerical_dict (DictConfig): Configuration mapping for
                                                columns to imputation strategy
                                                and parameters.

        Returns:
            pd.DataFrame: The dataframe with imputed numerical columns.
        """
        for column, impute_config in impute_numerical_dict.items():
            if impute_config.strategy != "constant":
                strategy = ImputationStrategy(impute_config.strategy)
                func = IMPUTATION_FUNCTIONS[strategy]
                if impute_config.strategy == ImputationStrategy.KNN:
                    dataframe = func(
                        dataframe,
                        column,
                        n_neighbors=impute_config.get("n_neighbors", 5),
                    )
                else:
                    dataframe = func(
                        dataframe,
                        column,
                    )
            else:
                dataframe[column].fillna(0.0, inplace=True)

            logging.info(f"Impute numerical column with {strategy} - {column}")

        return dataframe

    @staticmethod
    def _process_buildingsize(
        dataframe: pd.DataFrame, column: str, config: DictConfig
    ) -> pd.DataFrame:
        """
        Processes the building size column in the dataframe based on gross floor
        area and provided threshold.

        Args:
            dataframe (pd.DataFrame): The input dataframe to process.
            column (str): The column name representing building size.
            config (DictConfig): Configuration containing the threshold value or
                                 rules to calculate it.

        Returns:
            pd.DataFrame: The dataframe with processed building size column.
        """
        if not config.threshold:
            small_max = dataframe[dataframe[column] == "Small"][
                "grossfloorarea"
            ].max()
            large_min = dataframe[dataframe[column] == "Large"][
                "grossfloorarea"
            ].min()
            threshold = (small_max + large_min) / 2
        else:
            threshold = config.threshold

        dataframe.loc[
            (dataframe[column] == "Omit")
            & (dataframe["grossfloorarea"] <= threshold),
            column,
        ] = "Small"
        dataframe.loc[
            (dataframe[column] == "Omit")
            & (dataframe["grossfloorarea"] > threshold),
            column,
        ] = "Large"

        return dataframe

    @staticmethod
    def _process_greenmarkversion(
        dataframe: pd.DataFrame, column: str, config: DictConfig
    ) -> pd.DataFrame:
        """
        Processes the greenmark version column in the dataframe and sets
        specified values to 'Unknown'.

        Args:
            dataframe (pd.DataFrame): The input dataframe to process.
            column (str): The column name representing greenmark version.
            config (DictConfig): Configuration containing the value to match
                                 for replacing with 'Unknown'.

        Returns:
            pd.DataFrame: The dataframe with processed greenmark version column.
        """
        dataframe.loc[
            dataframe["greenmarkversion"] == config.value, column
        ] = "Unknown"

        return dataframe

    @staticmethod
    def _remove_symbols(
        dataframe: pd.DataFrame, remove_symbols_dict: DictConfig
    ) -> pd.DataFrame:
        """
        Removes specified symbols from columns in the given dataframe.

        Args:
            dataframe (pd.DataFrame): The input dataframe to process.
            remove_symbols_dict (DictConfig): Configuration mapping for columns
                                              to symbols that need to be
                                              removed.

        Returns:
            pd.DataFrame: The dataframe with specified symbols removed from
                          columns.
        """
        for column, symbol in remove_symbols_dict.items():
            dataframe[column] = dataframe[column].str.replace(symbol, "")
            logging.info(f"Replaced symbols in column - {column}")

        return dataframe


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataPreprocessor class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None.
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run DataPreprocessor class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str: Status of DataPreprocessor class.
    """
    preprocessor_config = cfg.data_processing.data_preprocessor
    preprocessor = DataPreprocessor(preprocessor_config)
    for dataset, config in preprocessor_config.datasets.items():
        if config.enable_preprocessing:
            preprocessor.preprocess_data(dataset=dataset)

    preprocessor.merge_datasets(
        output_folderpath=preprocessor_config.general.output_folderpath,
        suffix=preprocessor_config.general.suffix,
        common_identifier=preprocessor_config.general.common_identifier,
    )

    return "Complete data preprocessing"


if __name__ == "__main__":
    run_standalone()
