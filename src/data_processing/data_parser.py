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
            None.
        """
        self.cfg = cfg
        self.expected_files = self.cfg.expected_files
        self.output_folderpath = self.cfg.output_folderpath

        missing_files = self.check_expected_files(
            self.expected_files, self.output_folderpath
        )
        if missing_files:
            self._initialise_webdriver()
            self.fetch_and_download(missing_files)

    def fetch_and_download(self, missing_files: list) -> None:
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
        url = self.cfg.url
        dataset_xpaths = self.cfg.dataset_xpaths
        filtered_dataset_xpaths = {
            k: v for k, v in dataset_xpaths.items() if k in missing_files
        }

        # Fetch url
        self.driver.get(url)

        # Navigate and download
        for _, xpath in filtered_dataset_xpaths.items():
            self._navigate_and_download_dataset(xpath)
            self.driver.get(url)

        # Quit driver
        self.driver.quit()

        # Shift downloaded files to output folder
        self._move_files_from_downloads(missing_files, self.output_folderpath)

    def _click_element(self, xpath: str) -> None:
        """
        Clicks the web element identified by the provided xpath.

        Args:
            xpath (str): Xpath of the web element that needs to be clicked.
        """
        element = self.driver.find_element(By.XPATH, xpath)
        element.click()

    def _initialise_webdriver(self) -> None:
        """
        Initialise Chrome Webdriver.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": os.path.join(
                    os.path.expanduser("~"), self.cfg.downloads_foldername
                ),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            },
        )
        self.driver = webdriver.Chrome(
            ChromeDriverManager().install(), options=chrome_options
        )

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
        downloads_foldername = self.cfg.downloads_foldername

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

    def _navigate_and_download_dataset(self, xpath: str) -> None:
        """
        Navigate through the web page and download the dataset associated with
        the provided xpath.

        Args:
            xpath (str): Xpath of the dataset checkbox on the web page.
        """
        self._click_element(self.cfg.download_file_button_xpath)
        self._click_element(xpath)
        self._click_element(self.cfg.download_button_xpath)
        time.sleep(0.5)

    @staticmethod
    def check_expected_files(
        expected_files: Union[list, None], output_folderpath: str
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
        if not expected_files:
            logging.info("No files are expected")
            return False

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
    DataParser(cfg.data_processing.data_parser)

    return "Complete parsing and downloading of data"


if __name__ == "__main__":
    run_standalone()
