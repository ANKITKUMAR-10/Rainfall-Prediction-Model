🌧️ Rainfall Prediction Using Machine Learning 

End-to-end machine learning project that predicts whether it will rain today in the Melbourne region using historical weather data.
The project focuses on real-world ML practices such as class imbalance handling, pipelines, cross-validation, and model comparison.

📌 Problem Statement

Accurately predicting rainfall is critical for:

Weather warnings

Agriculture planning

Logistics & operations

Risk mitigation

This project builds a binary classification model to predict RainToday (Yes/No) using meteorological features.

🧠 Project Overview

Cleaned and preprocessed real-world weather data

Engineered seasonal features from dates

Addressed class imbalance

Built scalable ML pipelines

Compared Random Forest vs Logistic Regression

Optimized models using GridSearchCV

Evaluated using business-relevant metrics (Recall & F1-score)

📂 Dataset

Source: Australian weather observations

Region: Melbourne & surrounding areas

Target Variable: RainToday (Yes / No)

🧾 Feature Types

Numerical: Temperature, Humidity, Pressure, Wind Speed, Rainfall

Categorical: Location, Wind Direction, Season

Engineered Feature: Season (derived from Date)

🔄 Data Preprocessing Pipeline

✔ Automatically detected feature types
✔ Prevented data leakage using pipelines
✔ Applied transformations only on training folds

🔧 Preprocessing Steps
Feature Type	Technique
Numerical	StandardScaler
Categorical	OneHotEncoder
Target	Binary classification
Raw Data → Train/Test Split → Preprocessing → Model → Evaluation

⚖️ Class Balance Analysis

Rain occurs far less frequently than no rain

Dataset is imbalanced

Accuracy alone is misleading

📌 Key Insight:

A model predicting “No Rain” every day would already achieve high accuracy.

🏗️ Model Architecture
🔗 Scikit-learn Pipeline

Ensures clean, reproducible training

Avoids data leakage

Enables easy model swapping

🤖 Models Trained
1️⃣ Random Forest Classifier

Captures non-linear relationships

Strong overall accuracy

Lower recall for rain events

2️⃣ Logistic Regression

Simple, interpretable

Better recall for rainfall

Handles imbalance with class_weight='balanced'

🔍 Hyperparameter Optimization

Used GridSearchCV with Stratified K-Fold Cross Validation

Why Stratified?
→ Maintains rain/no-rain ratio across folds

📊 Model Evaluation Metrics

We evaluated models using:

Accuracy

Precision

Recall (True Positive Rate)

F1-Score

Confusion Matrix

📈 Model Comparison (Key Interview Highlight)
Metric	Random Forest	Logistic Regression
Accuracy	~0.84	~0.61–0.70
Recall (Rain = Yes)	❌ Low	✅ High
False Negatives	High	Low
Handles Imbalance	Weak	Better
Best for Warnings	❌	✅

📌 Business Decision:

Logistic Regression is preferred for rainfall warnings because missing rain is costlier than false alarms.

🔥 Feature Importance (Random Forest)

Top drivers of rainfall prediction:

Humidity (9am & 3pm)

Cloud cover

Pressure difference

Rainfall yesterday

Seasonality

📊 (Bar chart plotted in notebook)

📉 Confusion Matrix Interpretation

Logistic Regression significantly reduces false negatives

Better suited for risk-sensitive predictions

🧪 Final Test Performance

Model evaluated on unseen test data

Metrics reflect real-world deployment readiness

🛠️ Tech Stack
Tool	Purpose
Python	Core language
Pandas / NumPy	Data processing
Scikit-learn	ML pipelines & models
Matplotlib / Seaborn	Visualization
Jupyter Notebook	Experimentation
🚀 How to Run
pip install -r requirements.txt
jupyter notebook RainFallPrediction.ipynb

📌 Key Learnings (Interview Gold)

Why F1-score > accuracy for imbalanced datasets

How pipelines prevent data leakage

Model choice depends on business impact

Recall matters more when false negatives are costly

Simple models can outperform complex ones in practice

🔮 Future Improvements

Try XGBoost / LightGBM

Time-series modeling

Weather station clustering

Cost-sensitive learning

Model monitoring & drift detection

👤 Author

Ankit Kumar
Aspiring Data Scientist | Python | ML | Analytics
