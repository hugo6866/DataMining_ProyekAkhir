import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("resto_preprocessedz.csv", delimiter=";")

features = ["District", "subdistrict", "Highest Price", "Lowest Price", "Average Price", 
            "Category_Aneka nasi", "Category_Ayam & bebek", "Category_Bakmie", 
            "Category_Bakso & soto", "Category_Barat", "Category_Cepat saji", 
            "Category_Chinese", "Category_India", "Category_Jajanan", "Category_Jepang", 
            "Category_Kopi", "Category_Korea", "Category_Martabak", "Category_Minuman", 
            "Category_Pizza & pasta", "Category_Roti", "Category_Sate", "Category_Seafood", 
            "Category_Sweets", "Category_Thailand", "Category_Timur Tengah"]
target = "Popularity"

X = df[features]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:\n', class_report)

with open('knn_model.pickle', 'wb') as f:
    pickle.dump(knn, f)

with open('scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)
