import pandas as pd

parkinsons_df = pd.read_csv('parkinsons.csv')
parkinsons_df= parkinsons_df.dropna ()
parkinsons_df.head()

input_features = ['PPE', 'DFA']  
output_feature = 'status' 
X = parkinsons_df[input_features]
y = parkinsons_df[output_feature]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = svc.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Accuracy: {accuracy}")
import joblib

joblib.dump(SVC, 'svc_model.joblib')
