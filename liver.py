import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
data = pd.read_csv('C:/Users/vijay/mini/Liver Patient Dataset (LPD)_train.csv', encoding='latin-1') 
data['Gender of the patient'] = data['Gender of the patient'].replace({'Male': 1, 'Female': 0})
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.fillna(data.mean(), inplace=True)
data['Result'] = data['Result'].replace({'2': 0})
X = data.drop('Result',axis=1)
y = data['Result'] 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier() 
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, model.predict(X_test))*100) 
import pickle 
pickle.dump(model, open("liver.pkl",'wb')) 