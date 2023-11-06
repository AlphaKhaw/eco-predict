<!-- <p align="center">
  <img src="https://github.com/AlphaKhaw/eco-predict/assets/87654386/56162f8e-5be6-492e-aec0-7f1f74e18463" alt="Eco-Predict Banner" width="100%">
</p> -->

<p align="center">
  <img src="https://github.com/AlphaKhaw/eco-predict/assets/87654386/cc492db0-103a-442a-b8e4-45a6a6dc2f47" alt="Eco-Predict Banner" width="100%">
</p>


# Eco-Predict: AI-Driven Energy Consumption Analysis

Eco-Predict employs artificial intelligence to estimate the Energy Usage Intensity (EUI) of buildings in Singapore. Utilizing building attributes, it offers a predictive insight into energy consumption, fostering energy efficiency and architectural advancement.

## Introduction

The escalating costs and environmental impact of energy consumption necessitate a strategic approach to energy management in buildings. Eco-Predict addresses this by enabling precise predictions of EUI, guiding stakeholders toward energy optimization and sustainable design.


## Dataset Overview

This dataset contains the building energy performance data collected through BCA’s Building Energy Submission System (BESS), under the legislation on Annual Mandatory Submission of Building Information and Energy Consumption Data for Section 22FJ ‘Powers to Obtain Information’ of Building Control Act.

*Source*: https://beta.data.gov.sg/collections/22/datasets/d_e86d8a219d0936dbb321ade068a381da/view

## Prerequisites

- Docker (for local deployment)
- AWS account (for AWS Cloud deployment using EC2)


## Local Setup

1. **Clone the Repository**

   ```
   git clone https://github.com/AlphaKhaw/eco-predict.git
   cd eco-predict
   ```

2. **Build and Run with Docker**

   ```
   docker build -t alphakhaw/eco-predict:latest .
   docker run -d -p 8000:8000 alphakhaw/eco-predict:latest
   ```

    Or, deploy using **Docker Compose**:

   ```
   docker-compose up
   ```
3. **Access the FastAPI server locally at:** `http://localhost:8000`

## Usage

### Streamlit Interface

- Navigate to the [Streamlit Web Application](https://alphakhaw-eco-predict-app-6a2wmt.streamlit.app/)
- Input respective building features values.
- Click the "Predict" button to get the predicted **Energy Usage Intensity (EUI)** value.

### API Endpoints

The application provides a variety of endpoints for energy consumption prediction:

- `POST /predict_one/`: For predicting the Energy Usage Intensity (EUI) based on a single set of building features.
- `POST /predict_many/`: For batch predictions, accepting multiple sets of building features.
- `POST /predict_csv/`: For predictions based on a CSV file input, ideal for processing numerous entries at once.

Detailed instructions for using these endpoints, including the required payload formats and example responses, are available in our [GitHub Wiki](https://github.com/alphakhaw/eco-predict/wiki/API-Endpoints).

Please visit the Wiki to get a comprehensive guide on how to interact with each endpoint.


<!-- - `POST /predict_one/`:

  - **Description**: Accepts a single set of building features and returns the predicted EUI.
  - **Payload**:
    ```json
    {
        "percentageusageofled": float,
        "energyuseintensity_year": float,
        "typeofairconditioningsystem_DistrictCoolingPlant": integer (1 or 0),
        "averagemonthlybuildingoccupancyrate": float
    }
    ```
  - **Response**:
    ```json
    {
        "prediction": float
    }
    ```

- `POST /predict_many/`:

  - **Description**: Accepts multiple sets of building features and returns the predicted EUI for each set.
  - **Payload**:
    ```json
    [
        {
            "percentageusageofled": float,
            "energyuseintensity_year": float,
            "typeofairconditioningsystem_DistrictCoolingPlant": integer (1 or 0),
            "averagemonthlybuildingoccupancyrate": float
        },
        ...
    ]
    ```
  - **Response**:
    ```json
    {
        "predictions": [float, float, ...]
    }
    ```

- `POST /predict_csv/`:

  - **Description**: Accepts a CSV file with building features and returns the predicted EUI for each entry in the CSV.
  - **Payload**: A CSV file with headers:
    ```
    percentageusageofled,energyuseintensity_year,typeofairconditioningsystem_DistrictCoolingPlant,averagemonthlybuildingoccupancyrate
    50.0,120.0,1,85.0
    ...
    ```
  - **Response**:
    ```json
    {
        "predictions": [float, float, ...]
    }
    ``` -->
