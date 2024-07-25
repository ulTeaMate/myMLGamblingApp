Customer Churn Prediction in the gambling industry

Introduction

This project focuses on predicting customer churn using verious ML models. The models used are

MLP - Multilayer Perceptron
SVM - Support Vector Machine
DT - Decision Trees
RF - Random Forests
GB - Gradient Boosting

Project requirements

- python packages and libraries
- python scripts
- jupyter notebook is used for the model training and data analysis
- dataset gambling.csv found in this repository

Setup and Installation
- download or copy the repository
- create and activate jupyter notebook or use anaconda navigator to run the repository
- install the required packages.


  Data preparation

  - RFM is used to create categorical features from customer numerical data
  - target variable is churn vs non churners
  - churn definition is DaysLastOrder > 120 days

  Hyperparameter
  the models are trained and tested for the optimal combination of hyperparameters via cross validation technique

  Training and Evaluation
  - the data is split in test size 0.15 and 0.20. evaluation metrics are Accuracy, Precision, Recall and F1 Score.
 
  for the final model selection, pick the best fitted model from the list of selected models.
  Evaluation metrics are output.
  further research suggested with better datasets and real world implementation