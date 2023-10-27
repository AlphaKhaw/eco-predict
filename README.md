# Eco-Predict: AI-Driven Energy Consumption Analysis

Eco-Predict is an AI-powered tool designed to predict the Energy Usage Intensity (EUI) of buildings based on specific building features. The underlying model is served using FastAPI and the application interface is built using Streamlit.

## Problem Statement

With rising energy costs and increasing awareness about environmental sustainability, it's crucial for building owners, managers, and architects to understand a building's energy consumption patterns. Predicting the EUI of a building based on its features allows stakeholders to make informed decisions regarding energy conservation measures and sustainable design principles.

## Target Users

Building Owners: Understand energy performance to reduce costs.
Architects & Designers: Design buildings with optimal energy performance in mind.
Facility Managers: Implement energy conservation measures effectively.

### Dataset Description


Source: https://beta.data.gov.sg/collections/22/datasets/d_e86d8a219d0936dbb321ade068a381da/view

This dataset contains the building energy performance data collected through BCA’s Building Energy Submission System (BESS), under the legislation on Annual Mandatory Submission of Building Information and Energy Consumption Data for Section 22FJ ‘Powers to Obtain Information’ of Building Control Act.

The dataset consists of various building features, including:

`percentageusageofled`: The percentage of lighting fixtures that are LED.
`energyuseintensity_year`: Energy use intensity for specific years (2017, 2018, 2019).
`typeofairconditioningsystem_DistrictCoolingPlant`: Whether the building uses a district cooling plant type of air conditioning system (1 for Yes, 0 for No).
`averagemonthlybuildingoccupancyrate`: Average monthly building occupancy rate as a percentage.

**Note**: Please refer to the various Exploratory Data Analysis (EDA) notebooks for detailed analysis.

## Features

- Predict Energy Usage Intensity (EUI) based on building features.
- Intuitive Streamlit interface for user inputs.
- FastAPI backend for model serving.
- Dockerized application for easy deployment.

## Prerequisites

- Docker
- Docker Compose (for local deployment)
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

3. **Using Docker Compose**

   Assuming you have a `docker-compose.yml` file set up for the service:

   ```
   docker-compose up
   ```

   This will bring up all the services defined in your `docker-compose.yml` file.

4. Access the FastAPI server locally at: `http://localhost:8000`


## Deployment to AWS EC2

1. Launch a new EC2 instance on AWS.

2. SSH into your EC2 instance:

   ```
   ssh -i path_to_your_key.pem ec2-user@your-ec2-ip-address
   ```

3. Install Docker on EC2:

   ```
   sudo yum update -y
   sudo yum install docker -y
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

4. Log out and log back in again for the group permissions to take effect.

5. Pull the Docker image:

   ```
   docker pull alphakhaw/eco-predict:latest
   ```

6. Run the container:

   ```
   docker run -d -p 8080:8000 alphakhaw/eco-predict:latest
   ```

7. Access the FastAPI server at: `http://your-ec2-ip-address:8080`

## Usage

### Streamlit Interface

- Navigate to the Streamlit URL -
- Input building features such as the percentage usage of LED, type of air conditioning system, average monthly building occupancy rate, and energy use intensities for 2017, 2018, and 2019.
- Click the "Predict" button to get the predicted EUI value.

### API Endpoints

- `POST /predict_one/`:

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
    ```
