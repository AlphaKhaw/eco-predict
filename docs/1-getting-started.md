Welcome to the "Getting Started" section of `Eco-Predict`'s documentation. This guide will walk you through the initial setup and installation to get your development environment ready. Let's begin by ensuring you have the necessary prerequisites.

## Prerequisites

Before you install `Eco-Predict`, you need to have the following tools and accounts set up on your machine:

- **Git**: Version control is crucial for collaboration. Install Git from [git-scm.com](https://git-scm.com/).
- **Conda**: We recommend using Conda for managing your environments. Install it from [Anaconda's website](https://www.anaconda.com/products/distribution).
- **Docker**: For containerization and consistency across environments, Docker is essential. Download it from [Docker's website](https://www.docker.com/get-started).
- **AWS Account**: This is optional, but if you plan on deploying to the cloud, sign up for an account at [aws.amazon.com](https://aws.amazon.com/).

## Installation Guide

Once you have the prerequisites ready, follow these steps to set up your environment:

1. **Set up the Conda Environment**
   - Open your terminal or command prompt.
   - Create a new Conda environment by running:
     ```
     conda env create -f eco-predict-conda-env.yaml
     ```
   - Activate the new environment with:
     ```
     conda activate eco-predict
     ```

2. **Set up the Pre-commit Hook**
   - Set up the hooks with:
     ```
     pre-commit install
     ```
   - Now, pre-commit will run automatically on `git commit`.

## File Structure

Understanding the file structure is vital for navigating and contributing to the project. Here's the tree structure of the file system:

```plaintext
eco-predict/
├── README.md
├── app.py
├── conf
│   └── base
├── data
│   ├── raw
│   ├── preprocessed
│   └── split
├── docker
│   └── eco-predict-inference.Dockerfile
├── docker-compose.yaml
├── eco-predict-conda-env.yaml
├── models
│   ├── results
│   └── weights
├── notebooks
│   ├── initial_eda.ipynb
│   ├── eda_2019.ipynb
│   ├── eda_2020.ipynb
│   └── model_experimentation.ipynb
├── requirements.txt
└── src
    ├── __init__.py
    ├── base
    ├── data_processing
    ├── enums
    ├── fastapi
    ├── hyperparameter_tuning
    ├── inference
    ├── model
    ├── pipelines
    ├── tests
    ├── training
    └── utils
```

Here is a description for each component in the `eco-predict` project file structure:

- `README.md`: This markdown file contains an overview of the project, including how to set it up, how to use it, and any other pertinent information for users or contributors.

- `app.py`: Python script that runs the frontend `Streamlit` application.

- `conf/base`: A directory containing configuration files for the project, which may include settings for different environments such as development, testing, and production.

- `data`: This directory is structured to hold data in various stages of processing:
  - `raw`: Holds the initial, unprocessed data files as they were collected or received.
  - `preprocessed`: Contains data that has been cleaned and transformed, ready for analysis or model training.
  - `split`: Contains the data that has been divided into training, validation, and test sets.

- `docker`:
  - `eco-predict-inference.Dockerfile`: A Dockerfile script to create a container image for running the model inference in an isolated environment.

- `docker-compose.yaml`: A YAML file for defining and running multi-container Docker applications.

- `eco-predict-conda-env.yaml`: A YAML file that specifies all the dependencies needed to recreate the project's Conda environment.

- `models`:
  - `results`: This directory may contain output results from model training, such as evaluation metrics or logs.
  - `weights`: Stores the trained model weights or checkpoints.

- `notebooks`: A collection of Jupyter notebooks used for exploratory data analysis (EDA) and model experimentation:
  - `initial_eda.ipynb`: A notebook for initial exploration of the data set.
  - `eda_2019.ipynb`, `eda_2020.ipynb`: Notebooks containing EDA for the respective years' data.
  - `model_experimentation.ipynb`: A notebook used for trying out different modeling approaches.

- `requirements.txt`: A text file listing the Python packages required for the project, which can be installed using `pip`.

- `src`: The source directory containing the project's Python modules:
  - `__init__.py`: An initialization script that can turn the `src` directory into a Python package.
  - `base`: Contain base classes or functions used across the project.
  - `data_processing`: Contains scripts for processing and preparing the data.
  - `enums`: Enumerations that define a set of named values for various uses throughout the project.
  - `fastapi`: Modules related to the FastAPI framework for creating a web API.
  - `hyperparameter_tuning`: Code for tuning the model's hyperparameters to improve performance using `Optuna`.
  - `inference`: Scripts and modules to load the model and make predictions on new data.
  - `model`: Contains the definition and implementation of the predictive model or models.
  - `pipelines`: Contain data, model and combined pipelines orchestration scripts using `Prefect`.
  - `tests`: Contains test cases and unit testing scripts to ensure code reliability.
  - `training`: Includes scripts and modules for training machine learning models.
  - `utils`: Utility functions and helpers that provide common functionality across the project.
