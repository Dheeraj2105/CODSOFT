import streamlit as st
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit UI
st.title("💳 Fraud Detection System")
st.write("Upload a transaction dataset to predict fraud.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.write("### Dataset Preview")
    st.write(df.head())

    # Splitting Features and Labels
    X = df.drop(columns=["Class"], errors="ignore")
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    @st.cache_resource
    def train_model():
        model = xgb.XGBClassifier(eval_metric="logloss")
        model.fit(X_train, y_train)
        return model

    model = train_model()

    # Predictions
    y_pred = model.predict(X_test)

    # Model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"✅ **Model Accuracy on Test Data: {accuracy:.2f}**")

    # Predict on full dataset
    df["Prediction"] = model.predict(X)
    df["Prediction"] = df["Prediction"].map({0: "Legit", 1: "Fraudulent"})

    # Show results
    st.write("### Prediction Results")
    st.write(df.head(20))

    # Fraud Transaction Count
    fraud_count = (df["Prediction"] == "Fraudulent").sum()
    st.write(f"🔴 **Detected Fraudulent Transactions: {fraud_count}**")

    # Download Button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions", csv, "fraud_predictions.csv", "text/csv")
