# -*- coding: utf-8 -*-
"""
Machine learning

@author: Calotriton
"""

"""
Preparation

"""
import os  
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, PowerTransformer, FunctionTransformer,
    MinMaxScaler, StandardScaler, Normalizer, Binarizer, RobustScaler, label_binarize)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer


from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, GridSearchCV,
    RandomizedSearchCV, StratifiedKFold, KFold, LeaveOneOut, ShuffleSplit, RepeatedKFold)
from sklearn.ensemble import (
    BaggingClassifier, GradientBoostingClassifier, StackingClassifier, 
    RandomForestClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_auc_score, roc_curve, auc, precision_score, recall_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow import get_logger
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from imblearn.over_sampling import SMOTE



# Costumizations
sns.set_style( 'darkgrid' )
np.set_printoptions( precision = 2 )
get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")
seed = 2023


"""
Data exploration and transformation
  
"""
# Working directory
os.chdir(r'C:\Users\User\Documents\Master Data Science\Machine learning\Tarea')  

# Read file and initial analysis
file_path = r'C:\Users\User\Documents\Master Data Science\Machine learning\Task\task_data25.xlsx'
data_original = pd.read_excel(file_path)
data = data_original
print(data.shape)

# Get column names and descriptive statistics
column_names = data.columns.tolist()
print(column_names)
data.head()
data.describe() 

# 1. Convert "Levy" variable to numeric, replacing "-" with 0.
data['Levy'] = data['Levy'].replace('-', '0').astype(float)

# 2. Remove "Turbo" from "Engine volume" column and convert to numeric.
data['Engine volume'] = data['Engine volume'].str.replace('Turbo', '').astype(float)

# 3. Remove "km" from "Mileage" column and convert to numeric.
data['Mileage'] = data['Mileage'].str.replace(' km', '').astype(float)

# Verify results
print(data[['Levy', 'Engine volume', 'Mileage']].head())

# Check how many "Engine volume" values are 0 (2) and remove them
count_zeros = data[data['Engine volume'] == 0].shape[0]
print(count_zeros)
data = data[data['Engine volume'] != 0]

# Search for null values
data.isna().sum()

# Search for duplicates
print(data.duplicated().sum()) 
data = data.drop_duplicates() 
print(data.duplicated().sum())
print(data.shape)

# Show count of unique categories for the 'Color' variable
print(data['Color'].value_counts())
data['Color'].value_counts().plot(kind='bar')
plt.title('Color Distribution')
plt.xlabel('Color')
plt.ylabel('Count')
plt.show()

print(f'Variable types {data.dtypes}')

# Convert 'Price' and 'Levy' columns to float
data['Price'] = data['Price'].astype(float)
data['Levy'] = data['Levy'].astype(float)

# Verify changes
print(f'Variable types after conversion: {data.dtypes}')

"""
Standardization and Data Conversion
"""
col_cat = data[['Manufacturer', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color']]
col_num = data[['Price', 'Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']]

# Standardize
scaler = preprocessing.StandardScaler().fit(col_num)
col_num_standardized = scaler.transform(col_num)
col_num_standardized = pd.DataFrame(col_num_standardized, columns=col_num.columns, index=col_num.index)
df_cleaned = pd.concat([col_num_standardized, col_cat], axis=1) 

# Recover names  
df_cleaned = df_cleaned.set_axis(['Price','Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags','Manufacturer', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color'], axis=1)
sns.pairplot(df_cleaned, hue="Color", palette="bright")
print(df_cleaned.isna().sum())

# Convert categorical variables to dummies 
df_cleaned_dummies = pd.get_dummies(df_cleaned, columns=['Manufacturer', 'Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel'], drop_first=True)
df_cleaned_dummies.head()

# Final dataset
X = df_cleaned_dummies.drop('Color', axis=1)
y = df_cleaned_dummies["Color"]


"""
SVM Linear and RBF Models
"""
# Convert the target variable to numeric values
le = LabelEncoder()
y = le.fit_transform(data["Color"])  

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Define hyperparameters for grid search for two kernels: linear and Gaussian
param_grid = [
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
]

# GridSearchCV to find the best model
grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# Best model found
best_model = grid_search.best_estimator_
print(f"Best model: {grid_search.best_params_}")

# Evaluation on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
print(classification_report(y_test, y_pred))

# Confusion matrix for predicted and actual classes
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Bagging on the best model found
bagging_model = BaggingClassifier(estimator=best_model, n_estimators=500, random_state=123,
                                  max_samples=0.45, max_features=1.0, bootstrap_features=False)
bagging_model.fit(X_train, y_train)
y_pred_bagging = bagging_model.predict(X_test)

# Evaluation of the model with Bagging
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
auc_bagging = roc_auc_score(y_test, y_pred_bagging)
print(f"Bagging Accuracy: {accuracy_bagging:.4f}, Bagging AUC: {auc_bagging:.4f}")

# Model scores on test and training data
print('Model test Score: %.3f, ' %bagging_model.score(X_test, y_test),
      'Model training Score: %.3f' %bagging_model.score(X_train, y_train))

# Final comparison
print("Differences between SVM alone and with Bagging:")
print(f"Accuracy: SVM = {accuracy:.4f}, Bagging = {accuracy_bagging:.4f}")
print(f"AUC: SVM = {auc:.4f}, Bagging = {auc_bagging:.4f}")

"""
Stacking Model
"""
# Base models
model_1 = RandomForestClassifier(random_state=42)
model_2 = LogisticRegression(random_state=42, max_iter=1000)
model_3 = KNeighborsClassifier()

# Evaluate the individual performance of each model
models = {'Random Forest': model_1, 'Logistic Regression': model_2, 'KNN': model_3}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# Create the stacking model
stacking_model = StackingClassifier(
    estimators=[('rf', model_1), ('lr', model_2), ('knn', model_3)],
    final_estimator=SVC(probability=True),
    cv=5, passthrough=True
)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Evaluate the performance of stacking
y_pred_stack = stacking_model.predict(X_test)
stacking_acc = accuracy_score(y_test, y_pred_stack)
print("Stacking Model - Accuracy:", stacking_acc)
print(classification_report(y_test, y_pred_stack))

# Cross-validation
cv_results = cross_val_score(stacking_model, X_train, y_train, cv=5)
print(f"Cross-validation - Mean accuracy: {cv_results.mean():.4f}")
