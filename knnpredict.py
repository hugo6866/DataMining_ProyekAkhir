import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt

# Load the trained model and the scaler
with open('knn_model.pickle', 'rb') as f:
    knn = pickle.load(f)

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

# New data for prediction
input_data = {
    'District': [11],
    'subdistrict': [67],
    "Highest Price": [1400000],
    "Lowest Price": [3000],
    "Average Price": [53614],
    'Category_Aneka nasi': [0],
    'Category_Ayam & bebek': [0],
    'Category_Bakmie': [0],
    'Category_Bakso & soto': [0],
    'Category_Barat': [0],
    'Category_Cepat saji': [0],
    'Category_Chinese': [0],
    'Category_India': [0],
    'Category_Jajanan': [0],
    'Category_Jepang': [0],
    'Category_Kopi': [0],
    'Category_Korea': [1],
    'Category_Martabak': [1],
    'Category_Minuman': [0],
    'Category_Pizza & pasta': [0],
    'Category_Roti': [0],
    'Category_Sate': [0],
    'Category_Seafood': [0],
    'Category_Sweets': [0],
    'Category_Thailand': [0],
    'Category_Timur Tengah': [0]
}

input_df = pd.DataFrame.from_dict(input_data)

# Scale the input data
input_df = scaler.transform(input_df)

# Make a prediction for the input data
input_pred = knn.predict(input_df)
input_pred_proba = knn.predict_proba(input_df)

print("The predicted popularity for the input data is:", input_pred[0])
print("The predicted probabilities for each class are:", input_pred_proba[0])

# Plot the predicted probabilities
plt.figure(figsize=(10, 5))
plt.bar(range(len(knn.classes_)), input_pred_proba[0])
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.title('Predicted Probabilities for Each Class')
plt.xticks(range(len(knn.classes_)), knn.classes_)
plt.show()
