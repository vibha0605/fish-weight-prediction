#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Print the current working directory
print(os.getcwd())

# Load the dataset
df = pd.read_csv(r'C:\Users\vibha\OneDrive - Durham College\Desktop\Course Work_AI\Second Semester\2004\Lab 5\Lab #5 attached files Jul 25, 2024 843 PM\Fish.csv')
print(df.head())

# One-hot encode the 'Species' column
df = pd.get_dummies(df, columns=['Species'], drop_first=True)

# Split the data into features and target variable
X = df.drop('Weight', axis=1)
y = df['Weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to the specified directory
joblib.dump(model, r'C:\Users\vibha\OneDrive - Durham College\Desktop\Course Work_AI\Second Semester\2004\Lab 5\Lab #5 attached files Jul 25, 2024 843 PM\fish_weight_model.pkl')

