import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Estonia Survival Predictor", page_icon="🛟", layout="centered")

# -----------------------------
# Core configuration
# -----------------------------
ALL_FEATURES = ["Age", "Sex_encoded", "Category_encoded", "Country_encoded"]

FALLBACK_DEFAULTS = {
    "Age": 40.0,
    "Sex_encoded": 1,      # 1 = Male, 0 = Female
    "Category_encoded": 0, # 0 = Passenger, 1 = Crew
    "Country_encoded": 0
}

SEX_MAP = {"Male": 1, "M": 1, "Man": 1, "Boy": 1,
           "Female": 0, "F": 0, "Woman": 0, "Girl": 0}

CATEGORY_MAP = {"Passenger": 0, "P": 0, "Crew": 1, "C": 1}

BUILTIN_COUNTRIES = [
    "Unknown", "Estonia", "Sweden", "Finland", "Latvia", "Lithuania",
    "Russia", "Germany", "Norway", "Denmark", "Poland", "United Kingdom",
    "Netherlands", "France", "Other (type manually)"
]

# -----------------------------
# Utilities
# -----------------------------
def build_template_csv():
    return pd.DataFrame({
        "Age": [35, 50],
        "Sex": ["Male", "Female"],
        "Category": ["Passenger", "Crew"],
        "Country": ["Sweden", "Estonia"]
    })

def normalize_sex(value):
    return SEX_MAP.get(str(value).strip().title(), FALLBACK_DEFAULTS["Sex_encoded"])

def normalize_category(value):
    return CATEGORY_MAP.get(str(value).strip().title(), FALLBACK_DEFAULTS["Category_encoded"])

def encode_country_series(country_series, defaults, meta):
    if meta and "country_map" in meta:
        cmap = meta["country_map"]
        return country_series.map(cmap).fillna(defaults["Country_encoded"]).astype(int)
    return pd.Series([defaults["Country_encoded"]] * len(country_series), index=country_series.index)

def safe_float(value, default):
    try:
        return float(value) if pd.notna(value) else default
    except:
        return default

def ensure_features(df, defaults, selected_features):
    for col in selected_features:
        if col not in df.columns:
            df[col] = defaults[col]
    return df[selected_features]

def encode_upload(df_raw, defaults, meta, selected_features):
    df = df_raw.copy()
    if "Sex" in df.columns and "Sex_encoded" in selected_features:
        df["Sex_encoded"] = df["Sex"].apply(normalize_sex)
    if "Category" in df.columns and "Category_encoded" in selected_features:
        df["Category_encoded"] = df["Category"].apply(normalize_category)
    if "Age" in df.columns and "Age" in selected_features:
        df["Age"] = df["Age"].apply(lambda x: safe_float(x, defaults["Age"]))
    if "Country" in df.columns and "Country_encoded" in selected_features:
        df["Country_encoded"] = encode_country_series(df["Country"], defaults, meta)
    return ensure_features(df, defaults, selected_features)

def predict_with_proba(model, X):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, proba

# -----------------------------
# Load model and metadata
# -----------------------------
@st.cache_resource
def load_model_and_meta():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

    model_path = os.path.join(models_dir, "best_model.pkl")
    meta_path = os.path.join(models_dir, "preprocess_meta.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please place 'best_model.pkl' inside the 'models/' folder.")
        st.stop()

    model = joblib.load(model_path)
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
    return model, meta

model, meta = load_model_and_meta()
DEFAULTS = FALLBACK_DEFAULTS.copy()
if meta and "defaults" in meta:
    DEFAULTS.update(meta["defaults"])

# -----------------------------
# Sidebar: feature selection
# -----------------------------
st.sidebar.header("Select Features to Use")
selected_features = st.sidebar.multiselect(
    "Pick the features to include",
    options=ALL_FEATURES,
    default=ALL_FEATURES
)

if not selected_features:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛟 Estonia Passenger Survival Predictor")
st.write("Predict survival using manual inputs or file upload. Missing fields are auto-filled.")

mode = st.radio("Choose input method:", ["Manual entry", "Upload file"], horizontal=True)

# -----------------------------
# Manual entry
# -----------------------------
if mode == "Manual entry":
    st.subheader("Enter passenger details")
    row_data = {}
    col1, col2 = st.columns(2)

    if "Age" in selected_features:
        row_data["Age"] = col1.slider("Age (years)", 0, 100, int(DEFAULTS["Age"]))

    if "Sex_encoded" in selected_features:
        sex_val = col2.selectbox("Sex", ["Male", "Female"], index=DEFAULTS["Sex_encoded"])
        row_data["Sex_encoded"] = normalize_sex(sex_val)

    if "Category_encoded" in selected_features:
        cat_val = col2.selectbox("Category", ["Passenger", "Crew"], index=DEFAULTS["Category_encoded"])
        row_data["Category_encoded"] = normalize_category(cat_val)

    if "Country_encoded" in selected_features:
        country_choice = st.selectbox("Country", BUILTIN_COUNTRIES)
        manual_country_text = ""
        if country_choice == "Other (type manually)":
            manual_country_text = st.text_input("Type country name", value="")
        if meta and "country_map" in meta:
            cmap = meta["country_map"]
            if country_choice == "Other (type manually)" and manual_country_text.strip():
                row_data["Country_encoded"] = int(cmap.get(manual_country_text.strip(), DEFAULTS["Country_encoded"]))
            else:
                row_data["Country_encoded"] = int(cmap.get(country_choice, DEFAULTS["Country_encoded"]))
        else:
            row_data["Country_encoded"] = DEFAULTS["Country_encoded"]

    row = pd.DataFrame([row_data])
    row = ensure_features(row, DEFAULTS, selected_features)

    if st.button("Predict survival"):
        y_pred, proba = predict_with_proba(model, row)
        label = int(y_pred[0])
        survived_prob = float(proba[0]) if proba is not None else None
        st.markdown("---")
        st.success("Prediction: Survived" if label == 1 else "Prediction: Did Not Survive")
        if survived_prob is not None:
            st.info(f"Estimated survival probability: {survived_prob:.2%}")
        st.caption("Probability is an estimate based on the model and available features.")

# -----------------------------
# File upload
# -----------------------------
else:
    st.subheader("Upload CSV or Excel")
    st.write("Include any of: Age, Sex, Category, Country. Missing fields are auto-filled.")

    template = build_template_csv()
    csv_data = template.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download input template", data=csv_data, file_name="estonia_input_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded:
        df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df_raw.head())

        X = encode_upload(df_raw, DEFAULTS, meta, selected_features)
        y_pred, proba = predict_with_proba(model, X)

        out = df_raw.copy()
        out["Predicted_Survived"] = y_pred.astype(int)
        if proba is not None:
            out["Survival_Probability"] = proba

        st.success("✅ Predictions complete")
        st.dataframe(out)

        csv_out = out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download predictions", data=csv_out, file_name="estonia_predictions.csv", mime="text/csv")

# -----------------------------
# Help section
# -----------------------------
with st.expander("ℹ️ How this app works"):
    st.markdown("""
- You can enter data manually or upload a file for bulk predictions.
- Missing fields are filled with smart defaults based on training data.
- Sidebar allows you to dynamically select which features to include.
- Ensure `best_model.pkl` and `preprocess_meta.pkl` are in the `models/` folder.
""")
