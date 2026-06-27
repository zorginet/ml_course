# Bank Marketing Term Deposit Predictor API

## 📋 Project description
This service has been developed to predict whether a bank customer will agree to open a term deposit following a marketing campaign (a telephone call). The project automates the sales department’s work, allowing it to focus on the most promising customers.

## 📊 Data and Task
* **Data:** Bank Marketing Dataset (demographic data of the client, socio-economic indicators at the time of the call, history of previous contacts).
* **Task:** Binary classification (Target 1 — client agrees, Target 0 — client declines).
* **Model:** LightGBM Classifier, wrapped in a Scikit-Learn Pipeline (includes preprocessing and encoding of categorical features).

## 🛠️ Technologies Used
* **Python 3.12**
* **FastAPI** & **Uvicorn** (creating REST API)
* **Pydantic** (validating input data and aliases for column names)
* **LightGBM** & **Joblib** (working with the ML model)
* **Docker** (containerizing the application)
* **Render** (hosting and deploying the service)

## 🌐 Public Endpoint (Testing)
You can test the model's performance through the interactive Swagger interface at the link below:
👉 **https://ml-course.onrender.com/docs**