![](UTA-DataScience-Logo.png)
# Detecting Online Transaction Fraud via Machine Learning

## Business Problem
Our business problem is about correctly identifying fraudulent online transactions. In machine learning terms, this translates to a supervised learning binary classification problem.

## Project Overview
Our goal is to design a machine learning model that has at least an 80% F1 score for predicting the fradulent class an a 90% F1 score for predicting the non-fraudulent class. The reason for choosing F1 score is that it balances precision and recall, so the cost of false positives and false negatives are both accounted for. 

We approached this problem by starting with a logistic regression model as our baseline, and then applying a neural network model and an XGBoost model. These two models are more advanced forms of machine learning which we chose because they would be able to capture the complex relationships between the numerous features of this sizeable dataset.

## Data
Our data is from the IEEE-CIS Fraud Detection Kaggle Challenge. It is a tabular dataset that has records of transactions and information about the amount, card type, and more in a six month timeframe. There are 590,540 transactions. The data is given in the form of two csv files, one named identity and the other named transaction. The transaction dataset holds all transactions and on set of features, while the identity dataset holds a subset of the transactions with another different set of features regarding network information and the like. In other words, the identity dataset holds a bit more information for some transactions in the transaction dataset. We merge these two tables on their common features: TransactionID.

https://www.kaggle.com/competitions/ieee-fraud-detection/data

## Data Preprocessing
Most of our data cleaning focused on figuring out how to handle the large number of missing values, particularly for certain columns. Our initial approach was to drop the columns that had over 70% missing values, which we considered to be too noisy. For the rest of the missing values, because most of our features had certain values for most of the columns that had the highest frequency, we performed mode imputation.

For feature engineering, our initial approach for the neural network model was to create new features to gain insights on the counting, timedelta, and match features already given.

For our XGBoost model, we chose to engineer even more features including target, frequency, and aggregation encoding.

## Exploratory Data Analysis
Our initial look at the data had us comparing the feature distributions of the fraudulent class vs the non-fraudulent class.

## Modeling Approach
Our baseline model was a logistic regression model. Because the classes were imbalanced, we chose to use SMOTE to handle this problem and add synthetic fraudulent transactions to help with the modeling.

For our advanced models, we created an ANN model and an XGBoost model.

## Model Training
Our training involved using the scikit-learn library to train the logistic regression and XGBoost models, and tensorflow for the ANN model.

For the neural network model, we adjusted the number of epochs trained to 100 due to the large size of the data.

## Results
Our metrics involved the classification report built into scikit learn to look at the different predictive power of the classes.

## Model Interpretation
In order to interpret our models, we created SHAP plots to help our explainability.

## Key Insights
Our first major insight was that advanced modeling was essential; a logistic regression model does not have the predictive power to handle the complexity of this dataset.

Another insight was that feature engineering was essential to getting the models to perform successfully.

## Conclusion
Machine learning techniques are a tools that are able to learn and predict fraudulent transactions accurately while being able to reduce the amount of false flags and false positives.

## Future Work
Some improvements include performing ensemble techniques to combine different well performing models into a slightly better performing model. This is how the winning team in the Kaggle competition was able to such a high ROC AUC score.

## How to Run

## Repository Structure

## Requirements
