import os
import sys
from unittest.mock import patch

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from data_processing.data_parser import DataParser


@pytest.fixture
def config() -> DictConfig:
    """
    Fixture to provide a sample configuration for testing.

    Returns:
        DictConfig: A sample configuration.
    """
    with initialize(
        config_path="../../conf/base",
        job_name="test_data_parser",
        version_base="1.1",
    ):
        cfg = compose(config_name="test_pipelines.yaml")

    return cfg


@pytest.fixture
def dataparser(config: DictConfig) -> DataParser:
    """
    Fixture to create an instance of the DataParser class for unit testing.

    Args:
        config (DictConfig): Hydra configuration YAML.

    Returns:
        DataParser: An instance of the DataParser class.
    """
    return DataParser(config)


def test_check_expected_files(dataparser: DataParser):
    mock_output_folder = "/mock/path"
    expected_files = ["file1.txt", "file2.txt"]

    # 1. Test when all files exist
    with patch("os.path.exists", return_value=True):
        result = dataparser._check_expected_files(
            expected_files, mock_output_folder
        )
        assert result == False

    # 2. Test when some files are missing
    with patch("os.path.exists", side_effect=[True, False]):
        result = dataparser._check_expected_files(
            expected_files, mock_output_folder
        )
        assert result == ["file2.txt"]

    # 3. Test when all files are missing
    with patch("os.path.exists", return_value=False):
        result = dataparser._check_expected_files(
            expected_files, mock_output_folder
        )
        assert result == expected_files
