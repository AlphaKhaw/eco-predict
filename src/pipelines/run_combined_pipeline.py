import logging

from prefect import flow, task

from src.data_processing.data_parser import run_standalone as run_data_parser
from src.data_processing.data_preprocessor import (
    run_standalone as run_data_preprocessor,
)
from src.data_processing.data_splitter import (
    run_standalone as run_data_splitter
)
from src.hyperparameter_tuning.hyperparameter_tuner import (
    run_standalone as run_hyperparameter_tuning,
)
from src.training.train import run_standalone as run_model_training

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


@task
def model_training():
    run_model_training()


@task
def hyperparameter_tuning():
    run_hyperparameter_tuning()


@flow(name="End-to-End Pipeline", log_prints=True)
def run_pipeline():
    parse_data()
    preprocess_data()
    split_data()
    model_training()
    hyperparameter_tuning()


if __name__ == "__main__":
    run_pipeline.serve(name="test-run")
