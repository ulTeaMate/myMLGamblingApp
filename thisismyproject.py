#!/usr/bin/env python
# coding: utf-8

# In[61]:


# Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Data Preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Machine Learning Models
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Dataset
from sklearn.datasets import load_iris

# Model Evaluation and Selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

# Other
import datetime as dt
import re
from sklearn.model_selection import ParameterGrid


# In[62]:


## ↓ Import Gambling Dataset ↓
df = pd.read_csv('gambling.csv')


# In[63]:


# view information about the dataset
df.info()


# In[64]:


# view dimensions of dataset
df.shape


# In[65]:


# let's preview the dataset

df.head()


# In[66]:


# view the column names of the dataframe

col_names = df.columns

col_names
     


# In[67]:


# remove leading spaces from column names

df.columns = df.columns.str.strip()


# In[68]:


# view column names again

df.columns


# In[69]:


# check for missing values in variables
df.isnull().sum()


# In[70]:


# Replacing null or empty values with 0 in all columns
df.fillna(0.0, inplace=True)


# In[71]:


# iterate over the columns of the DataFrame
for col in df.columns:
    # check if the column is of type "int"
    if df[col].dtype == int:
        # convert the column to "float"
        df[col] = df[col].astype(float)


# In[72]:


# view summary statistics in numerical variables
round(df.describe(),2)


# In[ ]:





# In[73]:


# Sample data creation for the purpose of this example
df = df = pd.read_csv('gambling.csv')

# Creating the RFM DataFrame
df_RFM = df[['customerId', 'sum_stakes_fixedodds', 'sum_stakes_liveaction', 'sum_bets_casino', 'sum_bets_liveaction', 'DaysLastOrder']].copy()
df_RFM['Recency'] = df_RFM['DaysLastOrder']
df_RFM['MonetaryValue'] = df_RFM['sum_stakes_fixedodds'] + df_RFM['sum_stakes_liveaction']
df_RFM['Frequency'] = df_RFM['sum_bets_liveaction'] + df_RFM['sum_bets_casino']

# Calculate quartiles for each RFM metric
quartiles = df_RFM.quantile(q=[0.20, 0.40, 0.60, 0.80])

# Define function to assign quartile scores
def assign_quartile_score(value, quartile):
    if value <= quartile[0.20]:
        return 1
    elif value <= quartile[0.40]:
        return 2
    elif value <= quartile[0.60]:
        return 3
    elif value <= quartile[0.80]:
        return 4
    else:
        return 5

# Assign quartile scores to each RFM metric
df_RFM['R'] = df_RFM['Recency'].apply(assign_quartile_score, args=(quartiles['Recency'],))
df_RFM['F'] = df_RFM['Frequency'].apply(assign_quartile_score, args=(quartiles['Frequency'],))
df_RFM['M'] = df_RFM['MonetaryValue'].apply(assign_quartile_score, args=(quartiles['MonetaryValue'],))

# Concatenate RFM scores to create a combined RFM score
df_RFM['RFM_score'] = df_RFM['R'].map(str) + df_RFM['F'].map(str) + df_RFM['M'].map(str)

# Define dictionary to map RFM scores to segments using regular expressions
rfm_segment_dict = {
    'Not_Fan': [r'[1][1-5]', r'[2][1-2]'],
    'Switchers': [r'[2][3-5]', r'[3][1-2]', r'[4-5][1-2]'],
    'Loyal': [r'[3][3-5]', r'[4-5][3]'],
    'Champions': [r'[4-5][4-5]']
}

# Function to get segment for a given RFM score
def get_segment(rfm_score):
    segment = None
    for seg, patterns in rfm_segment_dict.items():
        for pattern in patterns:
            if re.match(pattern, rfm_score):
                segment = seg
                break
        if segment is not None:
            break
    return segment

# Add RFM segment column to the dataframe
df_RFM['RFM_segment'] = df_RFM['RFM_score'].apply(get_segment)

# Merge the RFM data back into the original dataframe
df = pd.merge(df, df_RFM[['customerId', 'Recency', 'Frequency', 'MonetaryValue', 'RFM_segment']], how='left', on='customerId')

# Plotting the count plot
plt.figure(figsize=(10, 6))
sns.countplot(x='RFM_segment', data=df, palette='viridis')
plt.title('Count of Customers by RFM Segment')
plt.xlabel('RFM Segment')
plt.ylabel('Count of Customers')

# Show plot
plt.show()


# In[74]:


# Sample data creation for the purpose of this example
df = df = pd.read_csv('gambling.csv')

# Classify churners and non-churners based on DaysLastOrder
df['Churner'] = df['DaysLastOrder'] > 120
df['Churner'] = df['Churner'].map({True: 'Churner', False: 'Non-Churner'})

# Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(x='DaysLastOrder', y=df.index, hue='Churner', style='Churner', 
                data=df, palette='coolwarm', s=100, alpha=0.9, edgecolor=None)

# Labeling the plot
plt.title('Churners vs Non-Churners Based on Days Last Order > 120 Days')
plt.xlabel('Days Last Order')
plt.ylabel('Index')  # Using the index for the y-axis
plt.legend(title='Churn Status')

# Show plot
plt.show()


# In[75]:


# Get the data types of each column
column_types = df.dtypes

# Count the occurrences of each data type
column_type_counts = column_types.value_counts().reset_index()
column_type_counts.columns = ['Column Type', 'Count']

# Print the counts table
print(column_type_counts)

# Plotting the table
plt.figure(figsize=(6, 3))
plt.table(cellText=column_type_counts.values, colLabels=column_type_counts.columns, cellLoc='center', loc='center')
plt.axis('off')
plt.title('Count of Column Types')
plt.show()


# In[76]:


churn_counts = df['churn'].value_counts()

print(churn_counts)


# In[77]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Suppose df is your DataFrame containing the data

# Delete invalid columns
df = df.drop('customerId', axis=1)
df = df.drop('DaysLastOrder', axis=1)

# Identify data types
data_types = df.dtypes
print("\nData Types:")
print(data_types)

# Perform data cleaning and preprocessing
for column in df.columns:
    if df[column].dtype == 'object':  # Handle categorical columns
        # Convert string columns to lowercase
        df[column] = df[column].str.lower()

        # Replace missing values with the most frequent value
        mode = df[column].mode().iloc[0]
        df[column].fillna(mode, inplace=True)

# Use get_dummies to perform one-hot encoding for all categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

for column in df.columns:
    if df[column].dtype in ['float64', 'float32', 'int64', 'int32']:  # Handle numerical columns
        # Replace missing values with the mean
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)

        # Normalize numerical columns using Min-Max scaling
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    else:
        # Handle other data types as needed
        pass

# Set the target variable as 'Churn'
y = df['churn']
X = df.drop('churn', axis=1)

# Display the cleaned and preprocessed dataset
print("\nCleaned and Preprocessed Dataset:")
print(X.head())
print("\nTarget variable:")
print(y.head())

     


# In[78]:


# Display the loaded dataset
print("Loaded Dataset:")
print(df.head())

# Identify data types
data_types = df.dtypes
print("\nData Types:")
print(data_types)

# Perform data cleaning and preprocessing
for column in df.columns:
    if df[column].dtype == 'object':  # Handle categorical columns
        # Convert string columns to lowercase
        df[column] = df[column].str.lower()

        # Replace missing values with the most frequent value
        mode = df[column].mode().iloc[0]
        df[column].fillna(mode, inplace=True)

        # Check if the column is still a categorical variable
        if df[column].dtype == 'object':
            # One-hot encode categorical columns using OneHotEncoder
            encoder = OneHotEncoder()
            encoded_columns = pd.DataFrame(encoder.fit_transform(df[[column]]).toarray(),
                                           columns=[f"{column}_{val}" for val in encoder.categories_[0]])
            df = pd.concat([df, encoded_columns], axis=1)
            df.drop(column, axis=1, inplace=True)

    elif df[column].dtype in ['float64', 'float32', 'int64', 'int32']:  # Handle numerical columns
        # Replace missing values with the mean
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)

        # Normalize numerical columns using Min-Max scaling
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    else:
        # Handle other data types as needed
        pass

# Set the target variable as 'Churn'
y = df['churn']
X = df.drop('churn', axis=1)

# Display the cleaned and preprocessed dataset
print("\nCleaned and Preprocessed Dataset:")
print(X.head())
print("\nTarget variable:")
print(y.head())
     


# In[79]:


from sklearn.ensemble import GradientBoostingClassifier

models = {
    'MLP': MLPClassifier(),
    'SVM': SVC(),
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Split the data into training and testing sets
test_size = 0.2  # set the test size
random_state = 42  # set the random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Loop over all models and evaluate their performance
accuracies = []
for model_name, model in models.items():
    accuracy, precision, recall, f1 = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    accuracies.append(accuracy)
    print("--------------------")
    print("Model: ", model_name)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("--------------------")


# In[80]:


from sklearn.ensemble import GradientBoostingClassifier

models = {
    'MLP': MLPClassifier(),
    'SVM': SVC(),
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Split the data into training and testing sets
test_size = 0.15  # set the test size
random_state = 42  # set the random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Loop over all models and evaluate their performance
accuracies = []
for model_name, model in models.items():
    accuracy, precision, recall, f1 = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    accuracies.append(accuracy)
    print("--------------------")
    print("Model: ", model_name)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print("--------------------")


# In[81]:


# List to store the selected models
selected_models = []

# Calculate the mean accuracy
mean_accuracy = np.mean(accuracies)
print("Mean Accuracy: {:.2f}%".format(mean_accuracy * 100))

# Iterate over the model names and their corresponding accuracies
for model_name, accuracy in zip(models.keys(), accuracies):
    MAD = mean_accuracy - accuracy  # Calculate the difference from the mean accuracy
    print("--------------------")
    print("Model: ", model_name)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Difference from Mean Accuracy: {:.2f}%".format(MAD * 100))

    # Check if the difference from the mean accuracy exceeds the threshold of 0.025
    if MAD > 0.025:
        print("--------------------")
        print("Tag: Does not enter the next phase")
    else:
        selected_models.append(model_name)  # Add the model to the selected models list
    print("--------------------")

# Print the list of selected models
print("Selected Models:", selected_models)


# In[82]:


# Define the machine learning models
research_models = {'MLP': MLPClassifier(),
          'SVM': SVC(),
          'Decision Trees': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(),
          'Gradient Boosting' : GradientBoostingClassifier()}

models = {chave: valor for chave, valor in research_models.items() if chave in selected_models}

# Define hyperparameters for each model
model_params = {
    'MLP': {
        'hidden_layer_sizes': [(3,), (5,), (10,), (64,)],
        'max_iter': [2000],
        'alpha': [0.001, 0.01],
        'solver': ['sgd', 'adam'],
        'activation': ['relu', 'tanh'],
        'verbose': [False]
    },
    'SVM': {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['auto', 'scale']
    },
    'Random Forest': {
        'n_estimators': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'random_state': [None]
    },
    'Decision Trees': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'learning_rate': [0.01, 0.05, 0.1]
    }
}

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate selected models with cross-validation
best_params = {}

# Train and evaluate selected models with cross-validation
for model_name in models:
    model = models[model_name]
    params = model_params[model_name]
    best_accuracy = 0
    best_params[model_name] = None
    for param_set in ParameterGrid(params):
        clf = model.set_params(**param_set)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        accuracy = scores.mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params[model_name] = param_set
    clf = model.set_params(**best_params[model_name])
    clf.fit(X_train, y_train)
    acc, prec, rec, f1 = train_and_evaluate(clf, X_train, X_test, y_train, y_test)
    print("--------------------")
    print(f"Best hyperparameters for {model_name}: {best_params[model_name]}")
    print(f"Accuracy for {model_name}: {acc}")


# In[83]:


best_model = None
best_score = 0

# Instantiation and training of the models that are in the 'best_params' dictionary
for model, params in best_params.items():
    if model == 'Decision Trees':
        clf = DecisionTreeClassifier(**params)
    elif model == 'Random Forest':
        clf = RandomForestClassifier(**params)
    elif model == 'SVM':
        clf = SVC(**params)
    elif model == 'MLP':
        clf = MLPClassifier(**params)
    elif model == 'Gradient Boosting':
        clf = GradientBoostingClassifier (**params)
    else:
        raise ValueError(f"Unrecognized model: '{model}'.")

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Accuracy of the {model} model: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model = clf


# In[84]:


# Creation of a dictionary with the best parameters of each model
selected_models_evaluation = best_params

# Select the model for evaluation
print("Select the model for evaluation:")
for model in selected_models_evaluation.keys():
    print(f"- {model}")

# ask the user to choose a template
chosen_model = input("Model: ")

# Verification if the chosen model is in the dictionary of best parameters
if chosen_model in best_params:
    # Instantiation and training of the chosen model
    params = best_params[chosen_model]
    if chosen_model == 'Decision Trees':
        clf = DecisionTreeClassifier(**params)
    elif chosen_model == 'Random Forest':
        clf = RandomForestClassifier(**params)
    elif chosen_model == 'SVM':
        clf = SVC(**params)
    elif chosen_model == 'MLP':
        clf = MLPClassifier(**params)
    elif chosen_model == 'Gradient Boosting':
        clf = GradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    # Evaluation of the chosen model and printing of the accuracy
    score = clf.score(X_test, y_test)

    # Verification if the chosen model is the best so far
    if score > best_score:
        best_score = score
        best_model = clf
else:
    raise ValueError(f"Unrecognized model: '{chosen_model}'.")

y_pred = clf.predict(X_test)
     


# In[85]:


cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix using predicted and true labels

print('Confusion matrix:\n\n', cm)  # Print the confusion matrix

tn, fp, fn, tp = cm.ravel()  # Unpack the elements of the confusion matrix into separate variables

print('\nTrue Positives(TP) =', tp)  # Print the number of true positives

print('True Negatives(TN) =', tn)  # Print the number of true negatives

print('False Positives(FP) =', fp)  # Print the number of false positives

print('False Negatives(FN) =', fn)  # Print the number of false negatives


# In[86]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
     


# In[87]:


# Generate a classification report by comparing the true labels (y_test) with the predicted labels (y_pred)
report = classification_report(y_test, y_pred)

print(report)  # Print the classification report
     


# In[88]:


TP = cm[0, 0]  # Assign the value at the top-left position of the confusion matrix to TP
TN = cm[1, 1]  # Assign the value at the bottom-right position of the confusion matrix to TN
FP = cm[0, 1]  # Assign the value in the first row and second column of the confusion matrix to FP
FN = cm[1, 0]  # Assign the value in the second row and first column of the confusion matrix to FN


# In[89]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# In[90]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# In[91]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# In[92]:


# print recall score

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# In[93]:


# print true_positive_rate score

true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
     


# In[94]:


# print false_positive_rate score

false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
     


# In[95]:


# print specificity score

specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[96]:


# plot ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a Churn')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()
     


# In[97]:


# compute ROC AUC

ROC_AUC = roc_auc_score(y_test, y_pred)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

