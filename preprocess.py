import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

#loading the dataset
df = pd.read_csv('cbc information 2.csv')

#inspecting the data
print(df.head())
print(df.columns)

#dropping id column
df = df.drop(['ID'], axis=1)

# Example target column; replace 'Disease' with the actual target column if available
# If you don't have a target column, you need to add one or adjust accordingly
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

#Data normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)


joblib.dump(scaler, 'scaler.pkl')

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Save the model
joblib.dump(model, 'disease_detection_model.pkl')

#Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
