# ml-task3-linear-regression
Overview:-This project implements Simple & Multiple Linear Regression using Python.
The goal is to understand how linear models work, how to preprocess data, how to train the model, and how to evaluate it using standard regression metrics.

🎯 Objectives
Import and preprocess the dataset
Split data into training and testing sets
Fit a Linear Regression model using scikit-learn
Evaluate model using MAE, MSE, RMSE, R²
Visualize regression results
Interpret coefficients
🧰 Tools & Libraries
Python
Pandas
NumPy
Scikit-learn
Matplotlib
📂 Dataset
You can use any regression dataset such as:
House Price Prediction dataset
California Housing dataset (from scikit-learn)
Your own custom dataset
In this project, the dataset is loaded either from scikit-learn or from a local CSV.

🚀 Steps Performed
1️⃣ Import & Preprocess the Dataset

Load dataset
Handle missing values
Feature scaling (using StandardScaler)

2️⃣ Train–Test Split
Split the dataset into:
80% training data
20% testing data

3️⃣ Train the Model

Fitted a Linear Regression model using:
from sklearn.linear_model import LinearRegression
Model Evaluation

Computed:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
RMSE (Root Mean Squared Error)
R² Score (coefficient of determination)

5️⃣ Visualizations
Predicted vs Actual plot
Residuals plot
Residuals histogram
Coefficient bar chart

📈 Results

The model provides insights into:
Which features contribute most
How well the regression line fits the data
Error distribution across predictions
(Results vary depending on dataset.)

📁 Project Structure
├── linear_regression.py / notebook.ipynb   # Main code
├── dataset.csv                             # Dataset (optional)
├── linear_regression_coefficients.csv      # Output file
├── linear_regression_evaluation.csv        # Metrics
└── README.md                               # Documentation

How to Run
Install dependencies:
pip install pandas numpy scikit-learn matplotlib
Run the script:
python linear_regression.py
Or open the .ipynb notebook and run all cells.
This project was completed as part of a Machine Learning learning task.

