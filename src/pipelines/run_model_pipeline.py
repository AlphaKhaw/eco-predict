import logging

from prefect import flow, task

from src.hyperparameter_tuning.hyperparameter_tuner import (
    run_standalone as run_hyperparameter_tuning,
)
from src.training.train import run_standalone as run_model_training

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


@task
def model_training():
    run_model_training()


@task
def hyperparameter_tuning():
    run_hyperparameter_tuning()


@flow(name="Model Pipeline", log_prints=True)
def run_pipeline():
    model_training()
    hyperparameter_tuning()


if __name__ == "__main__":
    run_pipeline.serve(name="test-run")
