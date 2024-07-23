import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#loading dataset
data = pd.read_csv('blood_data.csv')

#data cleaning
data.dropna(inplace=True)

#feature selection
X = data[['RBC', 'WBC', 'Hemoglobin', 'Platelets']]  # Example features
y = data['Disease']  # Target variable

#data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
