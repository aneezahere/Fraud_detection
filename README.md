# Fraud_detection_app 

A machine learning-based fraud detection system using balanced XGBoost model with real-time transaction analysis and batch processing capabilities

## link to the app 

https://fraud-detection.replit.app/

## Overview

This fraud detection system achieves:
- ROC-AUC Score: 0.9768
- Precision for Fraud Detection: 0.64
- Recall for Fraud Detection: 0.65
- Balanced handling of highly imbalanced dataset (0.58% fraud cases)

## Key Features

- **Dataset Analysis**: Support for both training dataset and new dataset uploads
- **Advanced Model**: Balanced XGBoost model optimized for fraud detection
- **Visual Analytics**: Interactive visualizations of risk distributions
- **AI-Powered Explanations**: Detailed transaction analysis using Groq AI
- **Batch Processing**: Ability to analyze multiple transactions simultaneously

## Model Architecture

The system uses a carefully tuned XGBoost model with:
- Scale position weight adjustment for imbalanced data
- Feature engineering including:
  - Time-based features (hour, day_of_week, is_night)
  - Location-based features (distance calculations)
  - Amount-based features (amount per capita)
  - Encoded categorical variables

## Getting Started

Follow these steps to set up and run the fraud detection system on your local machine.


Clone the Repository
Start by cloning the repository from GitHub:

bash
Copy code
git clone https://github.com/yourusername/fraud-detection-system.git

  Install Dependencies
Navigate to the project directory and install the required dependencies listed in requirements.txt:

bash
Copy code
cd fraud-detection-system
pip install -r requirements.txt

 Set Up Environment Variables
Add your Groq API key to the environment to enable AI-powered explanations. Replace your_api_key_here with your actual key.

bash
Copy code
export GROQ_API_KEY=your_api_key_here

 Run the Application
To start the Streamlit application, use the following command:

bash
Copy code streamlit run main.py

## Data Requirements

The system expects transaction data with the following features:

Transaction Details: amount, date/time
Location Information: latitude, longitude
Merchant Details: name, category
Customer Information: gender, job
Model Training Process

Data Preprocessing:

Handling missing values
Feature engineering and categorical encoding

Class Imbalance Handling:

SMOTE for minority class oversampling
Adjusted class weights for better fraud detection

Model Optimization:

Hyperparameter tuning
Cross-validation for model stability
Optimized for performance metrics tailored to imbalanced data

## Performance Metrics
ROC-AUC: 0.9768

Precision (Fraud): 0.64

Recall (Fraud): 0.65

F1-Score: 0.65

Overall Accuracy: 0.99

## License

This project is licensed under the MIT License - see the LICENSE file for details














