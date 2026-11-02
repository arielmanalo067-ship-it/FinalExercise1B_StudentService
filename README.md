# Student Service Request Monitoring System

## Project Overview
This Streamlit application monitors student service requests, analyzes request data, and predicts potential Service Level Agreement (SLA) breaches using a trained machine learning model.

## Objectives
- Monitor student service requests and processing delays.
- Analyze service request data and workload by office.
- Predict potential SLA breaches.
- Support staff review and decision-making.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Joblib

## Project Files
- `app.py` — Main Streamlit application.
- `requirements.txt` — Required Python packages.
- `student_service_requests_cleaned.csv` — Cleaned dataset.
- `models/trained_model.joblib` — Saved trained model.

## How to Run Locally
1. Install Python.
2. Open a terminal in the project folder.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Start the application:

   `streamlit run app.py`

## Important Limitation
Predictions are intended to support staff review. They should not replace authorized staff decisions. Prediction reliability depends on the quality of the dataset and the trained model.

## Deployment
The application is intended for deployment using GitHub and Streamlit Community Cloud.
