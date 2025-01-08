# Customer Churn Prediction Web Application

## Project Overview:
This project is a **Customer Churn Prediction** web application built using **Flask**, **Scikit-Learn**, and **Python**. 
The objective is to predict whether a customer will **churn** (stop doing business here bank subscription stoped) based on their demographic and behavioral data. 
The prediction is made using a **Decision Tree Classifier**, with **One-Hot Encoding (OHE)** for categorical data and **StandardScaler** for numerical feature scaling.

---

## Features:
- **Machine Learning Model:** Decision Tree Classifier  
- **Model Evaluation:** Confusion Matrix, Accuracy  
- **Preprocessing:**  
   - One-Hot Encoding for categorical features  
   - Standard Scaling for numerical features  
- **Web Interface:** Built using Flask  
- **Predictive Model Deployment:** Predicts if a customer has churned or not  
- **Dataset:** Bank Marketing Dataset (from UCI Machine Learning Repository)

---

## Project Structure:
```plaintext
├── templates/
│   ├── home.html          # Frontend HTML for input and result display
├── churn_model.pkl        # Trained Decision Tree Model (Saved using Pickle)
├── encoder.pkl            # One-Hot Encoder Object (Saved using Pickle)
├── scaler.pkl             # Standard Scaler Object (Saved using Pickle)
├── app.py                 # Flask Application File (Main Backend Logic)
├── README.md              # Project Documentation (This file)
├── bank.csv               # Dataset used for training and testing
├── requirements.txt       # Required Python Libraries
```
---

## Dataset Information:
Source: Bank Marketing Dataset (UCI Repository)
Target Variable: y (Binary classification: Yes/No indicating customer churn)
Key Features:
Age, Job, Marital Status, Education, Balance, Housing Loan, Duration, Campaign, etc.

