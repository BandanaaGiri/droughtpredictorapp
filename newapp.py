import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

st.set_page_config(page_title="Drought Predictor", layout="wide")


st.sidebar.image("logo.png", width=300)  

LABEL_MAP = {
    0: 'No Drought',
    1: 'Abnormally Dry',
    2: 'Moderate Drought',
    3: 'Severe Drought',
    4: 'Extreme Drought',
    5: 'Exceptional Drought'
}

FEATURE_COLUMNS = [
    'fips', 'PS', 'QV2M', 'T2MDEW', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE', 'TS',
    'WS10M_RANGE', 'WS50M', 'WS50M_RANGE', 'year', 'lat', 'lon', 'GRS_LAND'
]



@st.cache_resource(show_spinner="Loading model and scaler...")
def load_artifacts():
    scaler = joblib.load('scaler_selected_features.pkl')
    model = joblib.load('rf_gridsearch.pkl')
    return scaler, model

try:
    scaler, model = load_artifacts()
    st.sidebar.success("✅ Model and scaler loaded successfully")
except Exception as e:
    scaler, model = None, None
    st.sidebar.error(f"Failed to load model or scaler: {e}")

page = st.sidebar.selectbox("📂 Navigate", [
    "Upload Data", "Prediction Results", 
    "Feature Importance", "Feature Correlation Heatmap"
])

def read_file(file):
    try:
        return pd.read_csv(file)
    except Exception:
        return pd.read_excel(file)

def show_metrics(df):
    if "score" in df.columns:
        valid = df[df["score"].isin(LABEL_MAP.keys())]
        if valid.empty:
            st.info("No valid true labels available to compute metrics.")
            return
        y_true = valid["score"]
        y_pred = valid["prediction"]

        st.subheader("📈 Evaluation Metrics")
        st.write(f"🎯 **Accuracy:** `{accuracy_score(y_true, y_pred):.4f}`")
        st.write(f"📌 **Precision (Weighted):** `{precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}`")
        st.write(f"📈 **Recall (Weighted):** `{recall_score(y_true, y_pred, average='weighted'):.4f}`")
        st.write(f"📊 **F1 Score (Weighted):** `{f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}`")

        cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_MAP.keys()))
        st.subheader("🧮 Confusion Matrix (List Format)")
        st.code(str(cm.tolist()))
    else:
        st.info("True labels (`score` column) not found — evaluation metrics and confusion matrix unavailable.")

# --- Upload Data Tab ---
if page == "Upload Data":
    st.header("📄 Upload Your Data File")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file and model and scaler:
        df = read_file(uploaded_file)

        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        df = df.dropna(subset=FEATURE_COLUMNS)
        X = df[FEATURE_COLUMNS]
        X_scaled = scaler.transform(X)
        df["prediction"] = model.predict(X_scaled)
        df["drought_label"] = df["prediction"].map(LABEL_MAP)

        st.success("✅ Predictions complete!")

        st.session_state.df = df
        st.session_state.predicted = True

        show_metrics(df)

        st.subheader("📋 Prediction Results Preview")
        st.dataframe(df[["prediction", "drought_label"] + FEATURE_COLUMNS].head(50))

        st.download_button(
            label="📁 Download Full Results as CSV",
            data=df.to_csv(index=False).encode(),
            file_name="drought_predictions.csv",
            mime="text/csv"
        )
    elif uploaded_file and (model is None or scaler is None):
        st.warning("Model and scaler not loaded correctly.")
    else:
        st.info("Upload your input data file to get started.")

# --- Prediction Results Tab ---
elif page == "Prediction Results":
    st.header("Prediction Results and Filtering")

    if "df" in st.session_state and st.session_state.get("predicted", False):
        df = st.session_state["df"]

        selected_label = st.selectbox("Filter by Drought Severity", ["All"] + list(LABEL_MAP.values()))

        if selected_label != "All":
            filtered = df[df["drought_label"] == selected_label]
        else:
            filtered = df

        st.dataframe(filtered[[*FEATURE_COLUMNS, "prediction", "drought_label"]].head(50))
    else:
        st.info("Run predictions first in 'Upload Data' tab.")

# --- Feature Importance Tab ---
elif page == "Feature Importance":
    st.header("Feature Importance")

    if model and hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', ax=ax, color='teal')
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("Model does not provide feature importance.")

# --- Feature Correlation Heatmap Tab ---
elif page == "Feature Correlation Heatmap":
    st.header("Feature Correlation Heatmap")

    if "df" in st.session_state:
        df = st.session_state["df"]
        corr = df[FEATURE_COLUMNS].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='magma', fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap of Features")
        st.pyplot(fig)
    else:
        st.info("Run prediction first to see correlation heatmap.")

# --- Feedback and footer ---
st.sidebar.markdown("---")
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Your feedback")
if st.sidebar.button("Submit"):
    st.sidebar.success("Thanks for your feedback!")

st.markdown("---")
st.markdown("© 2025 | Developed by Bandana Giri | Version 1.0")
