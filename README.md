# ML-Ops Application
## Group 11

This repository contains code which demonstrates ML-Ops using a `FastAPI` application which predicts the defects class using the bug_pred.csv 

## Running Instructions
- Install dependencies using `pip3 install -r requirements.txt`
- Run application using `python3 main.py`
- Run tests using `pytest`

## CI/CD
- `build` (test) for all the pull requests
- `build` (test) and `upload_zip` for all pushes

## Group_11_Bug_Prediction_Application_Visualizatio_Explainability.ipynb
- Contains the working ML Model with Visualization and Explainability
- Visualization is done using Seaborn library/package
- Explainability is done using LIME and h2o AutoML

## main.py
- Implements the FastAPI

## ml_utils.py
- Implements the ML model

## test_app.py
- Implements usage of pytest
- It helps us in checking the CI/CD process( whether it passes or fails)
