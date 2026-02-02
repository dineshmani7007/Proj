import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svc import SVC
from sklearn.preprocessing import StandardScaler

# 1. Load your data
df = pd.read_csv("rugby_data.csv")

# 2. Prepare features (X) and target (y)
# Assuming 'Score_diff' is your feature and 'Result' (1 for Win, 0 for Loss) is target
X = df[['Score_diff']]
y = df['Result'] 

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Initialize and Train models
dt = DecisionTreeClassifier().fit(X_scaled, y)
rf = RandomForestClassifier().fit(X_scaled, y)
svc = SVC(probability=True).fit(X_scaled, y)

# 5. Save (Pickle) the models and the scaler
with open("DecisionTree_model.pkl", "wb") as f:
    pickle.load(dt, f)
with open("RandomForest_model.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("SVC_model.pkl", "wb") as f:
    pickle.dump(svc, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Models trained and saved successfully!")
