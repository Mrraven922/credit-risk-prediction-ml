# 💳 Credit Risk Prediction Platform – Machine Learning

Enterprise-grade Credit Risk Modeling Platform built using Python, Scikit-learn, and Streamlit, designed to evaluate customer creditworthiness using supervised machine learning techniques.

This project simulates a real-world banking risk analytics system, commonly used by financial institutions to minimize loan defaults and support data-driven credit approval decisions.

## Executive Summary

Credit risk assessment is a critical function within banking and financial services, directly impacting portfolio health, regulatory compliance, and profitability.

This solution implements a machine learning–based credit risk model trained on real-world financial data to classify customers as Low Risk (Good) or High Risk (Bad) borrowers.
The complete lifecycle is demonstrated — from data preprocessing and model training to deployment through an interactive web application.

## Business Objectives

Accurately assess customer credit risk

Reduce loan default probability

Support automated and data-driven credit decisions

Demonstrate production-ready ML deployment practices

Simulate real-world banking risk analytics workflows

## Solution Architecture
Data Source (CSV)

        ↓
        
Data Analysis & Feature Engineering (Jupyter)

        ↓
        
Categorical Encoding & Preprocessing

        ↓
        
Supervised Learning (Extra Trees Classifier)

        ↓
        
Model Serialization (Joblib)

        ↓
        

Interactive Risk Prediction App (Streamlit)



## Key Technologies
Layer	Technology

Programming	Python

Data Processing	Pandas, NumPy

Machine Learning	Scikit-learn

Model Algorithm	Extra Trees Classifier

Model Persistence	Joblib

Application Layer	Streamlit

Version Control	Git, GitHub


## Feature Set

The credit risk model evaluates customers using the following financial and demographic attributes:

Feature	Description

Age	Customer age

Sex	Gender category

Job	Employment type

Housing	Housing status

Saving Accounts	Savings account balance category

Checking Account	Checking account balance category

Credit Amount	Loan amount requested

Loan Duration	Loan tenure in months

## Model Development Details


Algorithm: Extra Trees Classifier

Learning Type: Supervised Classification

Target Variable: Credit Risk (Good / Bad)

Encoding Strategy: Label Encoding for categorical features

Model Persistence: Joblib serialization

Inference Mode: Single-record real-time prediction

## Why Extra Trees?

Handles complex non-linear relationships

Robust against overfitting

Well-suited for structured financial datasets

Strong performance without extensive hyperparameter tuning

Application Capabilities

Enterprise-style interactive UI

Real-time credit risk prediction

Consistent feature encoding between training and inference

## Clear classification output:

✅ Low Credit Risk (Good)

⚠️ High Credit Risk (Bad)


Production-aligned inference pipeline

Repository Structure

credit-risk-prediction-ml/

│
├── app.py                       # Streamlit inference application

├── Analysis_Model.ipynb         # EDA, preprocessing, model training

├── german_credit_data.csv       # Source dataset

├── extra_trees_model.pkl        # Trained ML model

├── Job_encoder.pkl              # Job encoder

├── Housing_encoder.pkl          # Housing encoder

├── Saving accounts_encoder.pkl  # Savings account encoder

├── Checking account_encoder.pkl # Checking account encoder

├── target_encoder.pkl           # Target label encoder

└── README.md                    # Project documentation


## Output

<img width="1876" height="900" alt="output" src="https://github.com/user-attachments/assets/071452f3-c2ee-40da-8f54-db9e31cabd57" />



(Streamlit-based Credit Risk Prediction Interface)
Interactive dashboard allowing users to input customer details and receive real-time credit risk classification.

## Setup & Execution

Environment Setup

python -m venv venv

source venv/bin/activate      # Linux / macOS

venv\Scripts\activate         # Windows


## Dependency Installation

pip install -r requirements.txt

## Application Execution
streamlit run app.py


The application will be accessible at:

http://localhost:8501

## Quality & Engineering Considerations

Reproducible machine learning pipeline

Clear separation between training and inference

Feature consistency enforcement

Scalable architecture for future enhancements

Maintainable and extensible codebase

Enterprise Use Cases

Bank loan approval systems

Credit card risk assessment

Retail and corporate lending analysis

Financial risk profiling

Decision support systems for underwriters

## Limitations & Assumptions

Dataset is static and CSV-based

Model retraining is manual

No real-time data ingestion

Regulatory constraints not explicitly modeled

## Future Enhancements

Model performance monitoring and drift detection

Database-backed data ingestion

REST API exposure for enterprise integration

Cloud deployment (AWS / Azure)

Explainable AI (SHAP) for model interpretability

Advanced analytics dashboards

## Author

Vignesh Raj
Machine Learning & Data Analytics Enthusiast

Focused on building production-ready, business-aligned machine learning solutions for banking and financial services.
