from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["ping"] == "pong" # Changing the syntax due to added timestamp


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Virginica" # Changing the syntax due to added timestamp


# Task 2: Added 2 Test Cases

def test_pred_versicolour():
    ''' Test case 1 '''

    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5,
        "sepal_width": 2.6,
        "petal_length": 3.8,
        "petal_width": 1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Versicolour" # Changing the syntax due to added timestamp

def test_pred_setosa():
    ''' Test case 2 ''' 
       
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5,
        "sepal_width": 4.3,
        "petal_length": 1,
        "petal_width": 0.8,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json()["flower_class"] == "Iris Setosa" # Changing the syntax due to added timestamp
