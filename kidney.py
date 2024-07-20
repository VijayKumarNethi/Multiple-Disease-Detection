import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
data = pd.read_csv(r'C:\Users\vijay\mini\kidney.csv') 
data.head() 
import seaborn as sns 
plt.figure(figsize=(10,10)) 
sns.heatmap(data.corr(), annot = True) 
X = data.drop('Class',axis=1)
y = data['Class'] 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier() 
model.fit(X_train, y_train) 
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test, model.predict(X_test))*100) 
import pickle 
pickle.dump(model, open("kidney.pkl",'wb')) 