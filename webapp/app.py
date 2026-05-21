# app.py
# XGB-based membrane performance prediction web app

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Membrane Performance Predictor - XGB",
    page_icon="🧪",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BUNDLE_FILES = {
    "NaCl rejection, %": "NaCl_rejection_pct_xgb_bundle.joblib",
    "MgCl2 rejection, %": "MgCl2_rejection_pct_xgb_bundle.joblib",
    "CaCl2 rejection, %": "CaCl2_rejection_pct_xgb_bundle.joblib",
    "Na2SO4 rejection, %": "Na2SO4_rejection_pct_xgb_bundle.joblib",
    "MgSO4 rejection, %": "MgSO4_rejection_pct_xgb_bundle.joblib",
    "Permeability, LMH/bar": "Permeability_LMH_bar_xgb_bundle.joblib",
}

DISPLAY_NAMES = {
    "NaCl rejection, %": "NaCl rejection (%)",
    "MgCl2 rejection, %": "MgCl₂ rejection (%)",
    "CaCl2 rejection, %": "CaCl₂ rejection (%)",
    "Na2SO4 rejection, %": "Na₂SO₄ rejection (%)",
    "MgSO4 rejection, %": "MgSO₄ rejection (%)",
    "Permeability, LMH/bar": "Permeability (LMH/bar)",
}


def load_bundles():
    bundles = {}

    for target, file_name in TARGET_BUNDLE_FILES.items():
        file_path = MODEL_DIR / file_name
        bundles[target] = joblib.load(file_path)

    return bundles


def get_all_input_columns(bundles):
    cols = []

    for bundle in bundles.values():
        for col in bundle["train_feature_cols"]:
            if col not in cols:
                cols.append(col)

    return cols


def default_for_column(col):

    col_lower = col.lower()

    if "type" in col_lower or "solvent" in col_lower:
        return "none"

    if "ratio" in col_lower:
        return 1.0

    if "ph" in col_lower:
        return 10.0

    if "pressure" in col_lower:
        return 15.0

    if "temperature" in col_lower:
        return 60.0

    if "time" in col_lower:
        return 60.0

    if "concentration" in col_lower:
        return 1000.0

    return 0.0


def is_categorical_column(col, bundles):

    for bundle in bundles.values():
        if col in bundle["cat_cols"]:
            return True

    return False


def build_input_ui(input_cols, bundles):

    st.sidebar.header("Input conditions")

    input_data = {}

    category_cols = [
        c for c in input_cols
        if is_categorical_column(c, bundles)
    ]

    numeric_cols = [
        c for c in input_cols
        if c not in category_cols
    ]

    with st.sidebar.expander("Categorical inputs", expanded=True):

        for col in category_cols:

            input_data[col] = st.text_input(
                col,
                value=str(default_for_column(col))
            )

    with st.sidebar.expander("Numeric inputs", expanded=True):

        for col in numeric_cols:

            input_data[col] = st.number_input(
                col,
                value=float(default_for_column(col)),
                step=0.01,
                format="%.2f"
            )

    return pd.DataFrame([input_data])


def predict_target(bundle, input_df):

    X = input_df.copy()

    for col in bundle["train_feature_cols"]:

        if col not in X.columns:

            if col in bundle["cat_cols"]:
                X[col] = "Missing"
            else:
                X[col] = 0

    X = X[bundle["train_feature_cols"]]

    for col, value in bundle["num_fill_map"].items():

        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            ).fillna(value)

    for col, value in bundle["cat_fill_map"].items():

        if col in X.columns:
            X[col] = X[col].astype(str).fillna(value)

    X_enc = bundle["encoder"].transform(X)

    for col in bundle["encoded_cols"]:

        if col not in X_enc.columns:
            X_enc[col] = 0

    X_enc = X_enc[bundle["encoded_cols"]]

    X_imp = bundle["imputer"].transform(X_enc)

    pred = bundle["model"].predict(X_imp)[0]

    return float(pred)


def main():

    st.title("XGB-based Membrane Performance Predictor")

    bundles = load_bundles()

    input_cols = get_all_input_columns(bundles)

    input_df = build_input_ui(input_cols, bundles)

    st.subheader("Current input")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    if st.button("Predict", type="primary"):

        predictions = {}

        for target, bundle in bundles.items():

            predictions[target] = predict_target(
                bundle,
                input_df
            )

        result_rows = []

        for target, value in predictions.items():

            result_rows.append({
                "Target": DISPLAY_NAMES[target],
                "Prediction": round(value, 3)
            })

        result_df = pd.DataFrame(result_rows)

        st.subheader("Prediction results")

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Permeability",
            f"{predictions['Permeability, LMH/bar']:.2f} LMH/bar"
        )

        save_df = input_df.copy()

        for target, value in predictions.items():
            save_df[target] = value

        save_path = OUTPUT_DIR / "latest_prediction.csv"

        save_df.to_csv(
            save_path,
            index=False,
            encoding="utf-8-sig"
        )

        st.success(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
