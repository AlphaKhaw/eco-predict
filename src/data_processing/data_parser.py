import logging
import os
import shutil
import time
from typing import Union

import hydra
from omegaconf import DictConfig
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class DataParser:
    """
    DataParser class to parse and download data from Data.gov.sg dataset.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a DataParser object.

        Args:
            cfg (DictConfig): Hydra configuration YAML.

        Returns:
            None
        """
        self.cfg = cfg
        self.expected_files = self.cfg.data_parser.expected_files
        self.output_folderpath = self.cfg.data_parser.output_folderpath
        missing_files = self._check_expected_files(
            self.expected_files, self.output_folderpath
        )
        if missing_files:
            chrome_options = Options()
            # chrome_options.add_argument("--headless")
            self.driver = webdriver.Chrome(
                ChromeDriverManager().install(), options=chrome_options
            )
            self._fetch_and_download(missing_files)

    def _fetch_and_download(self, missing_files: list) -> None:
        """
        Fetch the specified URL and download the datasets corresponding to the
        missing files using the provided xpaths.

        Args:
            missing_files (list): List of filenames corresponding to the
                                  datasets that need to be downloaded.

        Returns:
            None.
        """
        # Extract information from Hydra configuration YAML
        url = self.cfg.data_parser.url
        download_file_xpath = self.cfg.data_parser.download_file_button_xpath
        download_xpath = self.cfg.data_parser.download_button_xpath
        dataset_xpaths = self.cfg.data_parser.dataset_xpaths
        filtered_dataset_xpaths = {
            k: v for k, v in dataset_xpaths.items() if k in missing_files
        }

        # Fetch url
        self.driver.get(url)

        # Navigate and download
        for _, xpath in filtered_dataset_xpaths.items():
            download_file_button = self.driver.find_element(
                By.XPATH, download_file_xpath
            )
            download_file_button.click()
            dataset_checkbox = self.driver.find_element(By.XPATH, xpath)
            dataset_checkbox.click()
            download_button = self.driver.find_element(By.XPATH, download_xpath)
            download_button.click()
            time.sleep(0.5)
            self.driver.get(url)

        # Quit driver
        self.driver.quit()

        # Shift downloaded files to output folder
        self._move_files_from_downloads(missing_files, self.output_folderpath)

    def _move_files_from_downloads(
        self, missing_files: list, output_folderpath: str
    ) -> None:
        """
        Move the specified files from the Downloads directory to the desired
        output directory.

        Args:
            missing_files (list): List of filenames that need to be moved.
            output_folderpath (str): Path to the directory where files should
                                     be moved to.

        Returns:
            None.
        """
        # Get the user's home directory
        home = os.path.expanduser("~")

        # Get downloads folder
        downloads_foldername = self.cfg.data_parser.downloads_foldername

        # Iterate through list of missing files
        for filename in missing_files:

            # Build the path for the file in the Downloads directory
            source_path = os.path.join(home, downloads_foldername, filename)

            # Check if file exists in the source path
            if os.path.exists(source_path):

                # Build the destination path
                destination_path = os.path.join(output_folderpath, filename)

                # Move the file
                shutil.move(source_path, destination_path)

    @staticmethod
    def _check_expected_files(
        expected_files: list, output_folderpath: str
    ) -> Union[list, bool]:
        """
        Check for the presence of expected files in the specified directory.
        This static method evaluates the presence of a list of expected files
        in a provided directory path and returns a list of missing files if any,
        or False if all expected files are found.

        Args:
            expected_files (list): List of filenames that are expected to be in
                                   the directory.
            output_folderpath (str): Path to the directory where files are
                                     expected to be.

        Returns:
            Union[list, bool]:
            - list: A list of missing filenames if any files are missing.
            - bool: False if all files are present.
        """
        missing_files = [
            file
            for file in expected_files
            if not os.path.exists(os.path.join(output_folderpath, file))
        ]
        if missing_files:
            logging.info(
                f"Following files are missing: {', '.join(missing_files)}"
            )
            return missing_files

        logging.info("All expected files exist")
        return False


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone DataParser class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None.
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run DataParser class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str: Status of DataParser class.
    """
    DataParser(cfg)

    return "Complete parsing and downloading of data"


if __name__ == "__main__":
    run_standalone()
