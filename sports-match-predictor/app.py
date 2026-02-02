import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(page_title="Rugby Match Predictor", layout="centered")

st.title("Premiership Rugby 2025 — Match Predictor")
st.markdown("Predict the winner using **Decision Tree**, **Random Forest**, and **SVC** models.")

# ---------------------------------
# Function: Train models if missing
# ---------------------------------
def train_and_save_models(csv_file="rugby_data.csv"):
    df = pd.read_csv(csv_file)
    X = df[["Score_diff"]]
    y = df["Winner"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    svc_model = SVC(probability=True, random_state=42)
    svc_model.fit(X_train_scaled, y_train)

    # Save models
    with open("DecisionTree_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open("RandomForest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("SVC_model.pkl", "wb") as f:
        pickle.dump(svc_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("Models trained and saved successfully!")
    return df, dt_model, rf_model, svc_model, scaler

# ---------------------------------
# Function: Load models & data
# ---------------------------------
@st.cache_resource
def load_models(csv_file="rugby_data.csv"):
    required_files = ["DecisionTree_model.pkl", "RandomForest_model.pkl", "SVC_model.pkl", "scaler.pkl"]
    if all(os.path.exists(f) for f in required_files):
        with open("DecisionTree_model.pkl", "rb") as f:
            dt_model = pickle.load(f)
        with open("RandomForest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("SVC_model.pkl", "rb") as f:
            svc_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        df = pd.read_csv(csv_file)
        st.success("Models and data loaded successfully!")
        return df, dt_model, rf_model, svc_model, scaler
    else:
        st.warning("Model files not found. Training models now...")
        return train_and_save_models(csv_file)

# ---------------------------------
# Load models & data
# ---------------------------------
df, dt_model, rf_model, svc_model, scaler = load_models()

# ---------------------------------
# Show dataset
# ---------------------------------
with st.expander("View Dataset"):
    st.dataframe(df)

teams = sorted(set(df["Team_A"]).union(df["Team_B"]))

# ---------------------------------
# Match Prediction UI
# ---------------------------------
st.header("Predict a Match Result")
col1, col2 = st.columns(2)
team_a = col1.selectbox("Home Team", teams)
team_b = col2.selectbox("Away Team", teams)

if team_a == team_b:
    st.warning("Please select two different teams.")
    st.stop()

# ---------------------------------
# Feature Engineering
# ---------------------------------
team_matches = df[((df["Team_A"] == team_a) & (df["Team_B"] == team_b)) |
                  ((df["Team_A"] == team_b) & (df["Team_B"] == team_a))]

if team_matches.empty:
    team_a_avg = df[df["Team_A"] == team_a]["Score_diff"].mean()
    team_b_avg = df[df["Team_A"] == team_b]["Score_diff"].mean()
    avg_diff = 0 if pd.isna(team_a_avg) or pd.isna(team_b_avg) else (team_a_avg - team_b_avg)/2
    note = "No direct match data — using average team performance."
else:
    team_a_diff = team_matches.loc[team_matches["Team_A"] == team_a, "Score_diff"].mean()
    team_b_diff = -(team_matches.loc[team_matches["Team_A"] == team_b, "Score_diff"]).mean()
    avg_diff = (team_a_diff if pd.notna(team_a_diff) else 0) + (team_b_diff if pd.notna(team_b_diff) else 0)
    note = "Prediction based on past head-to-head data."

X_sample = pd.DataFrame({"Score_diff": [avg_diff]})
X_scaled = scaler.transform(X_sample)

# ---------------------------------
# Predictions
# ---------------------------------
models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "SVC": svc_model
}

predictions = {}
for name, model in models.items():
    pred = model.predict(X_scaled)[0]
    winner = team_a if pred == 1 else team_b
    predictions[name] = winner

# ---------------------------------
# Display Results
# ---------------------------------
st.subheader("Predicted Winners:")
for name, winner in predictions.items():
    st.write(f"**{name}** → {winner}")

st.info(f"Note: {note}")
st.markdown("---")
st.caption("Developed by Dinesh | Rugby Match Winner Predictor")
