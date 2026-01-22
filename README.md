# Heart Disease Prediction – Machine Learning Web Application

A comprehensive **Machine Learning–driven classification system** developed to predict the presence of heart disease using clinical and health-related attributes from the `heart.csv` dataset.

This project demonstrates the **complete lifecycle of a machine learning solution**, including data preprocessing, model development, algorithm comparison, performance evaluation, and deployment through a Flask-based web application.

---

## Project Overview

Heart disease remains one of the leading causes of mortality worldwide. Early prediction using patient health indicators can significantly support medical decision-making.

This project focuses on building a reliable predictive system by training and evaluating multiple machine learning classification algorithms and deploying the best-performing model as an interactive web application.

The application enables users to input patient health parameters and receive an instant prediction regarding the likelihood of heart disease.

---

## Objectives

* Develop an accurate machine learning model for heart disease prediction
* Perform comparative analysis of multiple classification algorithms
* Identify the best-performing model based on evaluation metrics
* Deploy the trained model using Flask
* Provide an easy-to-use and interactive prediction interface

---

## Dataset Description

The project utilizes the **heart.csv** dataset, which contains medical attributes commonly used for heart disease diagnosis.

Typical features include:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* Resting ECG results
* Maximum heart rate achieved
* Exercise-induced angina
* ST depression
* Slope of peak exercise ST segment
* Number of major vessels
* Thalassemia

**Target Variable:**

* Presence or absence of heart disease

---

## Machine Learning Models Implemented

Multiple classification models were developed and evaluated to understand their predictive behavior and performance.

### Models Used

* **K-Nearest Neighbors (KNN)**
  Distance-based classification algorithm that predicts outcomes based on nearest data points.

* **Logistic Regression**
  Linear model widely used for binary classification problems in healthcare.

* **Naive Bayes**
  Probabilistic classifier based on Bayes’ theorem with feature independence assumptions.

* **Decision Tree Classifier**
  Tree-based model that splits data using feature importance for decision making.

* **Random Forest Classifier**
  Ensemble technique combining multiple decision trees to improve accuracy and reduce overfitting.

* **AdaBoost Classifier**
  Boosting algorithm that converts weak learners into a strong classifier.

* **Gradient Boosting Classifier**
  Advanced ensemble model that minimizes prediction errors iteratively.

* **XGBoost Classifier**
  Optimized gradient boosting algorithm known for high performance and efficiency.

* **Support Vector Machine (SVC)**
  Margin-based classifier effective for high-dimensional feature spaces.

---

## Machine Learning Workflow

1. Data loading and initial exploration
2. Handling missing values and data cleaning
3. Feature encoding and transformation
4. Feature scaling using StandardScaler
5. Model training with multiple classifiers
6. Model evaluation using classification metrics
7. Selection of the best-performing model
8. Model serialization using pickle
9. Deployment through Flask web application

---

## Model Evaluation Metrics

The models were compared using the following performance metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

This comparative evaluation helped identify the most reliable model for deployment.

---

## Web Application Deployment

The selected machine learning model was deployed using **Flask**, enabling real-time predictions through a web interface.

### Application Features

* Interactive input form for patient health parameters
* Real-time heart disease prediction
* Input validation and error handling
* Lightweight and fast inference
* Clean and user-friendly interface

---

## Technology Stack

### Programming Language

* Python

### Machine Learning & Data Processing

* NumPy
* Pandas
* Scikit-learn
* XGBoost

### Backend Framework

* Flask

### Frontend

* HTML5
* CSS3

---

## Project Structure

```
heart-disease-prediction-ml/
│
├── app.py                     # Flask application
├── model.pkl                  # Trained ML model
├── scaler.pkl                 # Feature scaler
├── heart.csv                  # Dataset
│
├── templates/
│   └── index.html             # Web interface
│
├── static/
│   └── style.css              # Styling
│
└── README.md
```

---

## How to Run the Project Locally

### Step 1: Clone the Repository

```

```

### Step 2: Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```
pip install flask numpy pandas scikit-learn xgboost
```

### Step 4: Run the Application

```
python app.py
```

### Step 5: Open Browser

```
http://127.0.0.1:5000
```

---

## Use Cases

* Clinical decision support systems
* Healthcare analytics platforms
* Machine learning portfolio projects
* Academic research and learning
* Flask–ML deployment reference

---

## Future Enhancements

* Integration of SHAP for model explainability
* Probability-based risk scoring
* Cloud deployment (AWS, Render, Railway)
* Model monitoring and performance tracking
* Secure authentication for clinical environments

---

## Author

**Bachalakuri Ganesh**
Machine Learning & Python Developer

---

## Disclaimer

This project is intended strictly for educational and demonstration purposes and should not be used as a substitute for professional medical diagno
