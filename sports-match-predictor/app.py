# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Title ---
st.title("Sports Match Predictor")

# --- Upload CSV ---
st.sidebar.header("Upload Match Data CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    # --- Feature Selection ---
    st.sidebar.header("Select Features & Target")
    features = st.sidebar.multiselect("Select feature columns", df.columns.tolist())
    target = st.sidebar.selectbox("Select target column", df.columns.tolist())

    if st.sidebar.button("Train Model"):
        if len(features) == 0:
            st.error("Please select at least one feature column.")
        else:
            # --- Prepare Data ---
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- Train Model ---
            model = SVC()
            model.fit(X_train, y_train)

            # --- Predict & Evaluate ---
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model trained! Accuracy: {accuracy:.2f}")

            # --- Prediction Section ---
            st.subheader("Make a Prediction")
            input_data = {}
            for feature in features:
                input_data[feature] = st.number_input(f"Input value for {feature}", value=0)
            
            if st.button("Predict Match Outcome"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.write(f"Predicted Outcome: {prediction}")

else:
    st.info("Please upload a CSV file to get started.")
