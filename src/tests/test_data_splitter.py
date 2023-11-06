import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from omegaconf import OmegaConf

from src.data_processing.data_splitter import DataSplitter


class TestDataSplitter(unittest.TestCase):
    def setUp(self):
        self.config = OmegaConf.create(
            {
                "data_splitter": {
                    "general": {
                        "input_filepath": "path/to/input.csv",
                        "output_folderpath": "path/to/output",
                        "suffix": ["_train", "_val", "_test"],
                        "seed": 42,
                    },
                    "ratios": {
                        "train_ratio": 0.7,
                        "val_ratio": 0.15,
                        "test_ratio": 0.15,
                    },
                    "split_tolerance": 0.01,
                }
            }
        )

    @patch("src.data_processing.data_splitter.read_dataframe")
    @patch("src.data_processing.data_splitter.export_dataframe")
    @patch("src.data_processing.data_splitter.DataSplitter._check_input_ratio")
    @patch(
        "src.data_processing.data_splitter.DataSplitter._check_split_distribution"
    )
    @patch(
        "src.data_processing.data_splitter.DataSplitter._train_val_test_split"
    )
    def test_split_data(
        self,
        mock_split,
        mock_check_distribution,
        mock_check_ratio,
        mock_export,
        mock_read,
    ):
        # Set up mocks
        mock_read.return_value = MagicMock()
        mock_split.return_value = (MagicMock(), MagicMock(), MagicMock())

        # Instantiate DataSplitter and call split_data
        splitter = DataSplitter(self.config)
        splitter.split_data()

        # Assertions to ensure the functions are called with expected arguments
        mock_check_ratio.assert_called_once()
        mock_check_distribution.assert_called_once()
        mock_split.assert_called_once()

        # Assuming there's train, val, and test
        assert mock_export.call_count == 3

    def test_check_input_ratio(self):
        splitter = DataSplitter(self.config)
        with self.assertRaises(ValueError):
            splitter._check_input_ratio(
                train_ratio=0.6, val_ratio=0.1, test_ratio=0.2
            )
        with self.assertRaises(ValueError):
            splitter._check_input_ratio(
                train_ratio=0.6, val_ratio=1.0, test_ratio=0.4
            )
        splitter._check_input_ratio(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )

    def test_check_split_distribution(self):
        splitter = DataSplitter(self.config)
        mock_data = [pd.DataFrame() for _ in range(3)]
        for df in mock_data:
            df["data"] = range(10)

        with self.assertRaises(ValueError):
            splitter._check_split_distribution(
                split_datasets=mock_data, ratios=[0.6, 0.2, 0.2], tolerance=0.05
            )
        splitter._check_split_distribution(
            split_datasets=mock_data, ratios=[0.3, 0.3, 0.4], tolerance=0.1
        )

    def test_train_val_test_split(self):
        splitter = DataSplitter(self.config)
        mock_df = pd.DataFrame({"data": range(100)})

        train_data, val_data, test_data = splitter._train_val_test_split(
            dataframe=mock_df, ratios=[0.6, 0.2, 0.2], seed=42
        )
        self.assertEqual(len(train_data), 60)
        self.assertEqual(len(val_data), 20)
        self.assertEqual(len(test_data), 20)

        train_data, test_data = splitter._train_val_test_split(
            dataframe=mock_df, ratios=[0.7, 0.3], seed=42
        )
        self.assertEqual(len(train_data), 70)
        self.assertEqual(len(test_data), 30)


if __name__ == "__main__":
    unittest.main()
