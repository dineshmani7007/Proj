import streamlit as st
import pandas as pd
import pickle
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(page_title="Premiership Rugby 2025 Predictor", layout="centered")
st.title("Premiership Rugby 2025 — Match Predictor")
st.markdown("Predict the winner using **Decision Tree**, **Random Forest**, and **SVC** models.")

# ---------------------------------
# Step 0: Ensure CSV exists
# ---------------------------------
CSV_FILE = "rugby_data_report.csv"
MODEL_FILES = ["DecisionTree_model.pkl", "RandomForest_model.pkl", "SVC_model.pkl", "scaler.pkl"]

def fetch_and_prepare_data():
    """Fetch live JSON and create the processed CSV."""
    st.info("Fetching Premiership Rugby 2025 data from live feed...")
    url = "https://fixturedownload.com/feed/json/premiership-rugby-2025"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")
        st.stop()

    df = pd.DataFrame(data)
    df = df.dropna(subset=["HomeTeamScore", "AwayTeamScore"])
    df = df[df["HomeTeamScore"] != ""]
    df = df[df["AwayTeamScore"] != ""]
    df = df[["HomeTeam", "AwayTeam", "HomeTeamScore", "AwayTeamScore"]]
    df = df.rename(columns={
        "HomeTeam": "Team_A",
        "AwayTeam": "Team_B",
        "HomeTeamScore": "Score_A",
        "AwayTeamScore": "Score_B"
    })
    df["Score_A"] = df["Score_A"].astype(int)
    df["Score_B"] = df["Score_B"].astype(int)
    df["Score_diff"] = df["Score_A"] - df["Score_B"]
    df["Winner_flag"] = (df["Score_A"] > df["Score_B"]).astype(int)
    df.to_csv(CSV_FILE, index=False)
    st.success("Data fetched and CSV created successfully!")
    return df

# ---------------------------------
# Step 1: Train models if missing
# ---------------------------------
def train_and_save_models(df):
    X = df[["Score_diff"]]
    y = df["Winner_flag"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    svc_model = SVC(kernel='linear', probability=True, random_state=42)
    svc_model.fit(X_train, y_train)

    # Save models and scaler
    with open("DecisionTree_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open("RandomForest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("SVC_model.pkl", "wb") as f:
        pickle.dump(svc_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("Models trained and saved successfully!")
    return dt_model, rf_model, svc_model, scaler

# ---------------------------------
# Step 2: Load data and models
# ---------------------------------
@st.cache_resource
def load_data_and_models():
    # CSV
    if not os.path.exists(CSV_FILE):
        df = fetch_and_prepare_data()
    else:
        df = pd.read_csv(CSV_FILE)

    # Models
    missing_models = any(not os.path.exists(f) for f in MODEL_FILES)
    if missing_models:
        dt_model, rf_model, svc_model, scaler = train_and_save_models(df)
    else:
        with open("DecisionTree_model.pkl", "rb") as f:
            dt_model = pickle.load(f)
        with open("RandomForest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("SVC_model.pkl", "rb") as f:
            svc_model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    return df, dt_model, rf_model, svc_model, scaler

df, dt_model, rf_model, svc_model, scaler = load_data_and_models()

# ---------------------------------
# Step 3: Show dataset
# ---------------------------------
with st.expander("View Processed Data"):
    st.dataframe(df)

teams = sorted(set(df["Team_A"]).union(df["Team_B"]))

# ---------------------------------
# Step 4: Match Prediction UI
# ---------------------------------
st.header("Predict a Match Result")
col1, col2 = st.columns(2)
team_a = col1.selectbox("Home Team", teams)
team_b = col2.selectbox("Away Team", teams)

if team_a == team_b:
    st.warning("Please select two different teams.")
    st.stop()

# ---------------------------------
# Step 5: Feature Engineering
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
# Step 6: Predictions
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
# Step 7: Display Results
# ---------------------------------
st.subheader("Predicted Winners:")
for name, winner in predictions.items():
    st.write(f"**{name}** → {winner}")

st.info(f"Note: {note}")
st.markdown("---")
st.caption("Developed by Dinesh | Premiership Rugby 2025 Predictor")
