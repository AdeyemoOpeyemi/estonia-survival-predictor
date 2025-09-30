# app.py — Estonia Passenger Survival Predictor
# Requirements: streamlit, pandas, numpy, joblib, scikit-learn, os

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Estonia Survival Predictor", page_icon="🛟", layout="centered")

# -----------------------------
# Core configuration
# -----------------------------
EXPECTED_FEATURES = ["Age", "Sex_encoded", "Category_encoded", "Country_encoded"]

FALLBACK_DEFAULTS = {
    "Age": 40.0,
    "Sex_encoded": 1,      # 1 = Male, 0 = Female
    "Category_encoded": 0, # 0 = Passenger, 1 = Crew
    "Country_encoded": 0   # default/unknown bucket
}

SEX_MAP = {
    "Male": 1, "M": 1, "Man": 1, "Boy": 1,
    "Female": 0, "F": 0, "Woman": 0, "Girl": 0
}

CATEGORY_MAP = {
    "Passenger": 0, "P": 0,
    "Crew": 1, "C": 1
}

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

def encode_country_series(country_series, defaults, meta):
    if meta and "country_map" in meta:
        cmap = meta["country_map"]
        return country_series.map(cmap).fillna(defaults["Country_encoded"]).astype(int)
    return pd.Series([defaults["Country_encoded"]] * len(country_series), index=country_series.index)

def normalize_sex(value):
    v = str(value).strip().title()
    return SEX_MAP.get(v, FALLBACK_DEFAULTS["Sex_encoded"])

def normalize_category(value):
    v = str(value).strip().title()
    return CATEGORY_MAP.get(v, FALLBACK_DEFAULTS["Category_encoded"])

def safe_float(value, default):
    try:
        return float(value) if pd.notna(value) else default
    except:
        return default

def ensure_features(df, defaults):
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = defaults[col]
    return df[EXPECTED_FEATURES]

def encode_upload(df_raw, defaults, meta=None):
    df = df_raw.copy()
    df["Sex_encoded"] = df["Sex"].apply(normalize_sex) if "Sex" in df.columns else defaults["Sex_encoded"]
    df["Category_encoded"] = df["Category"].apply(normalize_category) if "Category" in df.columns else defaults["Category_encoded"]
    df["Age"] = df["Age"].apply(lambda x: safe_float(x, defaults["Age"])) if "Age" in df.columns else defaults["Age"]
    if "Country" in df.columns:
        df["Country_encoded"] = encode_country_series(df["Country"], defaults, meta)
    else:
        df["Country_encoded"] = defaults["Country_encoded"]
    return ensure_features(df, defaults)

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

    st.write("Looking for model at:", model_path)
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please place 'best_model.pkl' inside the 'models/' folder.")
        st.stop()

    # Load model using joblib
    model = joblib.load(model_path)

    # Optional metadata
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
    else:
        meta = None

    return model, meta

model, meta = load_model_and_meta()
DEFAULTS = FALLBACK_DEFAULTS.copy()
if meta and "defaults" in meta:
    DEFAULTS.update(meta["defaults"])

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
    st.caption("Uncheck a field to use default values.")

    col1, col2 = st.columns(2)
    with col1:
        use_age = st.checkbox("Age", value=True)
        age_val = st.slider("Age (years)", 0, 100, int(DEFAULTS["Age"])) if use_age else DEFAULTS["Age"]

        use_sex = st.checkbox("Sex", value=True)
        sex_readable = st.selectbox("Sex", ["Male", "Female"]) if use_sex else ("Male" if DEFAULTS["Sex_encoded"] == 1 else "Female")
        sex_val = normalize_sex(sex_readable)

    with col2:
        use_cat = st.checkbox("Category", value=True)
        cat_readable = st.selectbox("Category", ["Passenger", "Crew"]) if use_cat else ("Passenger" if DEFAULTS["Category_encoded"] == 0 else "Crew")
        cat_val = normalize_category(cat_readable)

        use_country = st.checkbox("Country", value=False)
        country_choice = "Unknown"
        manual_country_text = ""
        if use_country:
            if meta and "country_map" in meta:
                country_options = sorted(list(meta["country_map"].keys()))
                if "Unknown" not in country_options:
                    country_options = ["Unknown"] + country_options
                country_options = country_options + ["Other (type manually)"]
                country_choice = st.selectbox("Select Country", country_options)
                if country_choice == "Other (type manually)":
                    manual_country_text = st.text_input("Type country name", value="")
            else:
                country_choice = st.selectbox("Select Country", BUILTIN_COUNTRIES)
                if country_choice == "Other (type manually)":
                    manual_country_text = st.text_input("Type country name", value="")

    if use_country:
        if meta and "country_map" in meta:
            cmap = meta["country_map"]
            if country_choice == "Other (type manually)":
                typed = manual_country_text.strip()
                if typed:
                    country_encoded = int(cmap.get(typed, DEFAULTS["Country_encoded"]))
                    if typed not in cmap:
                        st.warning("Country not seen during training. Treated as Unknown.")
                else:
                    country_encoded = DEFAULTS["Country_encoded"]
            else:
                country_encoded = int(cmap.get(country_choice, DEFAULTS["Country_encoded"]))
        else:
            country_encoded = DEFAULTS["Country_encoded"]
            st.info("Country selection accepted, but without training-time encodings it won’t affect the prediction.")
    else:
        country_encoded = DEFAULTS["Country_encoded"]

    row = pd.DataFrame([{
        "Age": float(age_val),
        "Sex_encoded": int(sex_val),
        "Category_encoded": int(cat_val),
        "Country_encoded": int(country_encoded)
    }])
    row = ensure_features(row, DEFAULTS)

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

        X = encode_upload(df_raw, DEFAULTS, meta)
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
with st.expander("ℹ️ How this app handles Country and missing fields"):
    st.markdown("""
- Country dropdown is shown even if training metadata is missing. You can also type a custom country.
- If training-time country encodings (`preprocess_meta.pkl`) are available, the app uses them.
- If not available, Country is treated as Unknown internally to avoid mismatch with the trained model.
- You can enter data manually or upload a file; skipped fields use sensible defaults.
- Ensure `best_model.pkl` and `preprocess_meta.pkl` are in the `models/` folder at the same level as `apps/`.
""")
