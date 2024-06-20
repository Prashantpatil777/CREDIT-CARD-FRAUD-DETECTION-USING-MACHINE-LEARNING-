


"""CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING **PROJECT**

IMPORTING  LIBRARIES
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv("/content/creditcard.csv")

credit_card_data.head()

credit_card_data.tail()

credit_card_data.info()

credit_card_data.isnull().sum()

credit_card_data.shape

credit_card_data = credit_card_data.dropna(how='any')

credit_card_data.isnull().sum()

# Distribution of legit transaction and fraud transaction

credit_card_data['Class'].value_counts()

legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical measure of the data

legit.Amount.describe()

fraud.Amount.describe()

#compare the values for both the transactions

credit_card_data.groupby('Class').mean()

#under sampling method used ,build sample dataset

legit_sample = legit.sample(n=492)

#concanate two dataframes

new_dataset = pd.concat([legit_sample,fraud],axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

# spliting data into Features And Target

X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']

print(X)

print(Y)

# split the data into training and testing data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify =Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

"""Model Training

logistic regression for binary classification model

"""

# training the logistic regression model with train data

model = LogisticRegression()

model.fit(X_train,Y_train)

"""Model Evaluation

Accurancy Score
"""

#accurancy on training data

X_train_prediction = model.predict(X_train)
training_data_accurancy = accuracy_score(X_train_prediction,Y_train)

print('Accurancy on training data :',training_data_accurancy)

# accurancy on test data

X_test_prediction = model.predict(X_test)
test_data_accurancy = accuracy_score(X_test_prediction,Y_test)

print('Accurancy score on test data:',test_data_accurancy)

