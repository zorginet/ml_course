# 🌧️ Australian Rain Prediction App

This project demonstrates how to deploy a machine learning model using Streamlit to forecast the next day’s rainfall in Australia based on daily meteorological observations. The application allows users to analyse historical data and obtain an accurate probability-based forecast using a trained **Random Forest** model.

You can test the app by following this link: `link` 
*(If the message ‘This app has gone to sleep due to inactivity’ appears when you click the link, simply tap the ‘Yes, get this app back up!’ button and wait half a minute).*

---

## 📊 Project Description and Technologies
* **Task:** Binary classification (whether it will rain tomorrow: Yes/No).
* **Data:** Meteorological data from various stations in Australia (`weatherAUS.csv`).
* **Preprocessing:** Implemented full data processing pipeline (median/mean imputation of missing values, scaling of numerical features using `StandardScaler` and encoding of categorical variables using `OneHotEncoder`).
* **Technology Stack:** `Python`, `Streamlit`, `Scikit-Learn`, `Pandas`, `Numpy`, `Joblib`.

---

## 📁 Project Structure

* `data/`: Directory containing the dataset (`weatherAUS.csv`).
* `images/`: Directory for storing interface images (`australia.jpg`).
* `models/`: Directory containing the saved ML model along with preprocessing transformers (`aussie_rain.joblib`).
* `app.py`: Main Streamlit application file with interactive analytics charts.
* `requirements.txt`: List of required Python packages for deployment.

---

## 🛠️ Setup and Local Execution

### Prerequisites
Ensure you have Python 3.10 or a newer version installed.

### Installation and Execution
Copy and run these commands sequentially in your terminal:


1. Clone the repository to your computer
```bash
git clone https://github.com/zorginet/ml_course.git
```

2. Navigate to the folder with this specific project
```bash
cd module_5_deployment_and_mlops/australian-rain-predictor
```

3. Create a virtual environment (an isolated box for libraries)
```bash
python -m venv venv
```

4. Activate the virtual environment (command for macOS)
```bash
source venv/bin/activate
```

5. Install all required libraries from the requirements file
```bash
pip install -r requirements.txt
```

6. Run the Streamlit app locally
```bash
streamlit run app.py
```