import sys

import requests

sys.path.append("../../")

base_url = "http://0.0.0.0:8000"


def test_health():
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200
    print(response.json())


def test_predict_one():
    data = {
        "percentageusageofled": 50.0,
        "energyuseintensity_2017": 120.0,
        "typeofairconditioningsystem_DistrictCoolingPlant": 1.0,
        "averagemonthlybuildingoccupancyrate": 85.0,
        "energyuseintensity_2018": 115.0,
        "energyuseintensity_2019": 110.0,
    }

    response = requests.post(f"{base_url}/predict_one/", json=data)
    # assert response.status_code == 200
    print(response)
    # print(response.json())


def test_predict_many():
    data = {
        "data": [
            {
                "percentageusageofled": 50.0,
                "energyuseintensity_2017": 120.0,
                "typeofairconditioningsystem_DistrictCoolingPlant": 1.0,
                "averagemonthlybuildingoccupancyrate": 85.0,
                "energyuseintensity_2018": 115.0,
                "energyuseintensity_2019": 110.0,
            },
            {
                "percentageusageofled": 45.0,
                "energyuseintensity_2017": 125.0,
                "typeofairconditioningsystem_DistrictCoolingPlant": 0.0,
                "averagemonthlybuildingoccupancyrate": 90.0,
                "energyuseintensity_2018": 112.0,
                "energyuseintensity_2019": 108.0,
            },
        ]
    }

    response = requests.post(f"{base_url}/predict_many/", json=data)
    assert response.status_code == 200
    print(response.json())


def test_predict_csv():
    with open("data/split/test.csv", "rb") as file:
        files = {"file": ("test.csv", file)}
        response = requests.post(f"{base_url}/predict_csv/", files=files)
        # assert response.status_code == 200
        print(response.json())


if __name__ == "__main__":
    test_health()
    test_predict_one()
    test_predict_many()
    test_predict_csv()
