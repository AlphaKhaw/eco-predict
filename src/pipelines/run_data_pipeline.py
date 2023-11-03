import logging

from prefect import flow, task

from src.data_processing.data_parser import run_standalone as run_data_parser
from src.data_processing.data_preprocessor import (
    run_standalone as run_data_preprocessor
)
from src.data_processing.data_splitter import (
    run_standalone as run_data_splitter
)

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


@task
def parse_data():
    run_data_parser()


@task
def preprocess_data():
    run_data_preprocessor()


@task
def split_data():
    run_data_splitter()


@flow(name="Data Pipeline", log_prints=True)
def run_pipeline():
    parse_data()
    preprocess_data()
    split_data()


if __name__ == "__main__":
    run_pipeline.serve(name="test-run")
