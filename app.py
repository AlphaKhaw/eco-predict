import logging

import requests
import streamlit as st
import yaml


def input_page(endpoint: str):
    # Introduction Section
    st.title(
        "Building Energy Performance - Energy Usage Intensity (EUI) Predictor"
    )
    st.write(
        """
        Welcome to the EUI Predictor. This application allows you to input
        specific building features and get a prediction for the Energy Usage
        Intensity (EUI).

        Please provide values for the following features and click 'Predict' to
        see the estimated EUI.
        """
    )

    # User Input Section

    # Continuous input for percentage usage of led
    percentageusageofled = st.number_input(
        "Percentage Usage of LED:",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        format="%.2f",
    )

    # Yes or No input for type of air conditioning system
    ac_system = st.selectbox(
        'Is the Air Conditioning System of the "District Cooling Plant" type?',
        options=["Yes", "No"],
    )

    # Continuous input for average monthly building occupancy rate
    occupancy_rate = st.number_input(
        "Average Monthly Building Occupancy Rate (%):",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        format="%.2f",
    )

    # Continuous input for 2017, 2018 and 2019 energy use intensity
    energy_2017 = st.number_input("2017 Energy Use Intensity (EUI):", value=0.0)
    energy_2018 = st.number_input("2018 Energy Use Intensity (EUI):", value=0.0)
    energy_2019 = st.number_input("2019 Energy Use Intensity (EUI):", value=0.0)

    if st.button("Predict"):
        # Prepare the data for API request
        data = {
            "percentageusageofled": percentageusageofled,
            "energyuseintensity_2017": energy_2017,
            "typeofairconditioningsystem_DistrictCoolingPlant": 1.0
            if ac_system == "Yes"
            else 0.0,
            "averagemonthlybuildingoccupancyrate": occupancy_rate,
            "energyuseintensity_2018": energy_2018,
            "energyuseintensity_2019": energy_2019,
        }

        # Make request to FastAPI endpoint
        response = requests.post(f"{endpoint}/predict_one/", json=data)
        value = response.json()["prediction"]

        # Show results on Streamlit
        if response.status_code == 200:
            st.success(f"Predicted EUI: {value:.2f}")
        else:
            st.error("Error in API call")


if __name__ == "__main__":
    # Page Configurations
    st.set_page_config(
        page_title="Eco-Predict: AI-Driven Energy Consumption Analysis",
        layout="wide",
    )

    try:
        with open("conf/base/inference.yaml", "r") as stream:
            cfg = yaml.safe_load(stream)
            endpoint = cfg["streamlit"]["fastapi_endpoint"]
    except Exception as e:
        logging.info(f"An error occurred while loading the configuration: {e}")
        raise e

    input_page(endpoint)
