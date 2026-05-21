from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Membrane Performance Predictor", layout="wide")

st.title("Membrane Performance Predictor")
st.caption("Selected fabrication/operation inputs → five salt rejections and permeability prediction")


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

MODEL_DIR_CANDIDATES = [
    APP_DIR / "models",
    ROOT_DIR / "models",
]

DATA_PATH_CANDIDATES = [
    APP_DIR / "data" / "cleaned_with_retry_smiles.csv",
    ROOT_DIR / "data" / "cleaned_with_retry_smiles.csv",
    APP_DIR / "cleaned_with_retry_smiles.csv",
    ROOT_DIR / "cleaned_with_retry_smiles.csv",
]

OUTPUT_DIR_CANDIDATES = [
    APP_DIR / "outputs",
    ROOT_DIR / "outputs",
]


def first_existing_dir(candidates: List[Path], fallback: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return fallback


def first_existing_file(candidates: List[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


MODEL_DIR = first_existing_dir(MODEL_DIR_CANDIDATES, APP_DIR / "models")
OUTPUT_DIR = first_existing_dir(OUTPUT_DIR_CANDIDATES, APP_DIR / "outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = first_existing_file(DATA_PATH_CANDIDATES)


TARGET_BUNDLE_FILES = {
    "NaCl rejection, %": "NaCl_rejection_pct_xgb_bundle.joblib",
    "MgSO4 rejection, %": "MgSO4_rejection_pct_xgb_bundle.joblib",
    "Na2SO4 rejection, %": "Na2SO4_rejection_pct_xgb_bundle.joblib",
    "MgCl2 rejection, %": "MgCl2_rejection_pct_xgb_bundle.joblib",
    "CaCl2 rejection, %": "CaCl2_rejection_pct_xgb_bundle.joblib",
    "Permeability, LMH/bar": "Permeability_LMH_bar_xgb_bundle.joblib",
}

TARGET_DISPLAY_ORDER = [
    "NaCl rejection, %",
    "MgSO4 rejection, %",
    "Na2SO4 rejection, %",
    "MgCl2 rejection, %",
    "CaCl2 rejection, %",
    "Permeability, LMH/bar",
]

SELECTED_INPUT_COLS = [
    "Monomer A1 type",
    "Monomer A2 type",
    "A1/A2 ratio",
    "Monomer A concentration, wt%",
    "Monomer B type",
    "Monomer B concentration, wt%",
    "Organic solvent type",
    "Additive X1 type in aqueous phase",
    "Additive X1 concentration, wt%",
    "Additive X2 type in aqueous phase",
    "Additive X2 concentration, wt%",
    "Aqueous phase pH",
    "Additive Y type in organic phase",
    "Additive Y concentration, wt%",
    "Nanomaterials type in aqueous phase",
    "Nanomaterials loading in aqueous phase, wt%",
    "Nanomaterials type in organic phase",
    "Nanomaterials loading in organic phase, wt%",
    "Polymerization time, s",
    "Heat curing time, min",
    "Heat curing temperature, degree",
    "Transmembrane pressure, bar",
    "NaCl concentration, ppm",
    "MgSO4 concentration, ppm",
    "Na2SO4 concentration, ppm",
    "MgCl2 concentration, ppm",
    "CaCl2 concentration, ppm",
]

CATEGORICAL_INPUT_COLS = [
    "Monomer A1 type",
    "Monomer A2 type",
    "Monomer B type",
    "Organic solvent type",
    "Additive X1 type in aqueous phase",
    "Additive X2 type in aqueous phase",
    "Additive Y type in organic phase",
    "Nanomaterials type in aqueous phase",
    "Nanomaterials type in organic phase",
]

FLOAT_2_COLS = {
    "A1/A2 ratio",
    "Monomer A concentration, wt%",
    "Monomer B concentration, wt%",
    "Additive X1 concentration, wt%",
    "Additive X2 concentration, wt%",
    "Aqueous phase pH",
    "Additive Y concentration, wt%",
    "Nanomaterials loading in aqueous phase, wt%",
    "Nanomaterials loading in organic phase, wt%",
}

INTEGER_COLS = {
    "Polymerization time, s",
    "Heat curing time, min",
    "Heat curing temperature, degree",
    "Transmembrane pressure, bar",
    "NaCl concentration, ppm",
    "MgSO4 concentration, ppm",
    "Na2SO4 concentration, ppm",
    "MgCl2 concentration, ppm",
    "CaCl2 concentration, ppm",
}

DISPLAY_LABELS = {
    "MgSO4 concentration, ppm": "MgSO₄ concentration, ppm",
    "Na2SO4 concentration, ppm": "Na₂SO₄ concentration, ppm",
    "MgCl2 concentration, ppm": "MgCl₂ concentration, ppm",
    "CaCl2 concentration, ppm": "CaCl₂ concentration, ppm",
    "MgSO4 rejection, %": "MgSO₄ rejection, %",
    "Na2SO4 rejection, %": "Na₂SO₄ rejection, %",
    "MgCl2 rejection, %": "MgCl₂ rejection, %",
    "CaCl2 rejection, %": "CaCl₂ rejection, %",
    "Heat curing temperature, degree": "Heat curing temperature, °C",
}

CATEGORICAL_NONE_INTERNAL_VALUE = "Missing"


def label_text(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)


def short_target_label(target: str) -> str:
    return label_text(target).replace(" rejection, %", "")


def display_to_model_value(value: object) -> object:
    if isinstance(value, str) and value.strip().lower() == "none":
        return CATEGORICAL_NONE_INTERNAL_VALUE
    return value


def model_to_display_value(value: object) -> str:
    if pd.isna(value):
        return "None"
    value_str = str(value).strip()
    if value_str.lower() in {"missing", "nan", "none", "<na>", ""}:
        return "None"
    return value_str


def is_none_selected(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() == "none"


def load_dataset(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    return pd.DataFrame()


def load_bundles() -> tuple[Dict[str, dict], List[str]]:
    bundles = {}
    missing = []

    for target, file_name in TARGET_BUNDLE_FILES.items():
        path = MODEL_DIR / file_name
        if not path.exists():
            missing.append(str(path))
            continue
        bundles[target] = joblib.load(path)

    return bundles, missing


def infer_default_numeric(df: pd.DataFrame, col: str, bundles: Dict[str, dict] | None = None) -> float:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return float(s.median())

    if bundles:
        vals = []
        for bundle in bundles.values():
            if col in bundle.get("num_fill_map", {}):
                try:
                    vals.append(float(bundle["num_fill_map"][col]))
                except Exception:
                    pass
        if vals:
            return float(pd.Series(vals).median())

    return 0.0


def extract_encoder_categories(bundles: Dict[str, dict], col: str) -> List[str]:
    vals = []

    for bundle in bundles.values():
        encoder = bundle.get("encoder", None)

        ordinal_encoder = getattr(encoder, "ordinal_encoder", None)
        category_mapping = getattr(ordinal_encoder, "category_mapping", None)

        if not category_mapping:
            continue

        for item in category_mapping:
            if item.get("col") != col:
                continue

            mapping = item.get("mapping")
            if mapping is None:
                continue

            try:
                candidates = list(mapping.index)
            except Exception:
                candidates = []

            for v in candidates:
                label = model_to_display_value(v)
                if label not in {"-1", "-2"}:
                    vals.append(label)

    vals = sorted(set(v for v in vals if v and v != "None"))
    return vals


def infer_category_options(df: pd.DataFrame, col: str, bundles: Dict[str, dict]) -> List[str]:
    vals = []

    if col in df.columns:
        vals.extend([model_to_display_value(v) for v in df[col].drop_duplicates().tolist()])

    vals.extend(extract_encoder_categories(bundles, col))

    vals = sorted(set(v for v in vals if v and v != "None"))

    return ["None"] + vals


def numeric_input(col: str, default: float, disabled: bool = False):
    if disabled:
        default = 0.0

    if col in FLOAT_2_COLS:
        return st.number_input(
            label_text(col),
            min_value=0.0,
            value=round(float(default), 2),
            step=0.01,
            format="%.2f",
            disabled=disabled,
            key=f"input_{col}",
        )

    if col in INTEGER_COLS:
        return st.number_input(
            label_text(col),
            min_value=0,
            value=int(round(float(default))),
            step=1,
            format="%d",
            disabled=disabled,
            key=f"input_{col}",
        )

    return st.number_input(
        label_text(col),
        value=float(default),
        step=0.01,
        format="%.2f",
        disabled=disabled,
        key=f"input_{col}",
    )


def categorical_input(df: pd.DataFrame, col: str, bundles: Dict[str, dict]):
    opts = infer_category_options(df, col, bundles)

    if len(opts) <= 1:
        return st.text_input(label_text(col), value="None", key=f"input_{col}")

    return st.selectbox(label_text(col), opts, index=0, key=f"input_{col}")


def build_selected_input_panel(df: pd.DataFrame, bundles: Dict[str, dict]) -> Tuple[bool, Dict[str, object]]:
    values: Dict[str, object] = {}

    st.subheader("Input condition")
    st.caption("선택/입력 조건 기반 전체 target 동시 예측")

    tab_basic, tab_additive, tab_process, tab_salt = st.tabs(
        ["Monomer / Solvent", "Additive / Nanomaterials", "Process", "Salt / Pressure"]
    )

    with tab_basic:
        row1 = st.columns(2)
        with row1[0]:
            values["Monomer A1 type"] = categorical_input(df, "Monomer A1 type", bundles)
        with row1[1]:
            values["Monomer A2 type"] = categorical_input(df, "Monomer A2 type", bundles)

        row2 = st.columns(2)
        with row2[0]:
            values["A1/A2 ratio"] = numeric_input(
                "A1/A2 ratio",
                infer_default_numeric(df, "A1/A2 ratio", bundles),
                disabled=is_none_selected(values["Monomer A2 type"]),
            )
        with row2[1]:
            values["Monomer A concentration, wt%"] = numeric_input(
                "Monomer A concentration, wt%",
                infer_default_numeric(df, "Monomer A concentration, wt%", bundles),
            )

        row3 = st.columns(2)
        with row3[0]:
            values["Monomer B type"] = categorical_input(df, "Monomer B type", bundles)
        with row3[1]:
            values["Monomer B concentration, wt%"] = numeric_input(
                "Monomer B concentration, wt%",
                infer_default_numeric(df, "Monomer B concentration, wt%", bundles),
            )

        values["Organic solvent type"] = categorical_input(df, "Organic solvent type", bundles)

    with tab_additive:
        cols = st.columns(2)
        with cols[0]:
            values["Additive X1 type in aqueous phase"] = categorical_input(df, "Additive X1 type in aqueous phase", bundles)
            values["Additive X1 concentration, wt%"] = numeric_input(
                "Additive X1 concentration, wt%",
                infer_default_numeric(df, "Additive X1 concentration, wt%", bundles),
                disabled=is_none_selected(values["Additive X1 type in aqueous phase"]),
            )

            values["Additive X2 type in aqueous phase"] = categorical_input(df, "Additive X2 type in aqueous phase", bundles)
            values["Additive X2 concentration, wt%"] = numeric_input(
                "Additive X2 concentration, wt%",
                infer_default_numeric(df, "Additive X2 concentration, wt%", bundles),
                disabled=is_none_selected(values["Additive X2 type in aqueous phase"]),
            )

            values["Aqueous phase pH"] = numeric_input(
                "Aqueous phase pH",
                infer_default_numeric(df, "Aqueous phase pH", bundles),
            )

        with cols[1]:
            values["Additive Y type in organic phase"] = categorical_input(df, "Additive Y type in organic phase", bundles)
            values["Additive Y concentration, wt%"] = numeric_input(
                "Additive Y concentration, wt%",
                infer_default_numeric(df, "Additive Y concentration, wt%", bundles),
                disabled=is_none_selected(values["Additive Y type in organic phase"]),
            )

            values["Nanomaterials type in aqueous phase"] = categorical_input(df, "Nanomaterials type in aqueous phase", bundles)
            values["Nanomaterials loading in aqueous phase, wt%"] = numeric_input(
                "Nanomaterials loading in aqueous phase, wt%",
                infer_default_numeric(df, "Nanomaterials loading in aqueous phase, wt%", bundles),
                disabled=is_none_selected(values["Nanomaterials type in aqueous phase"]),
            )

            values["Nanomaterials type in organic phase"] = categorical_input(df, "Nanomaterials type in organic phase", bundles)
            values["Nanomaterials loading in organic phase, wt%"] = numeric_input(
                "Nanomaterials loading in organic phase, wt%",
                infer_default_numeric(df, "Nanomaterials loading in organic phase, wt%", bundles),
                disabled=is_none_selected(values["Nanomaterials type in organic phase"]),
            )

    with tab_process:
        cols = st.columns(3)
        process_cols = [
            "Polymerization time, s",
            "Heat curing time, min",
            "Heat curing temperature, degree",
        ]
        for i, col in enumerate(process_cols):
            with cols[i % 3]:
                values[col] = numeric_input(col, infer_default_numeric(df, col, bundles))

    with tab_salt:
        cols = st.columns(2)
        salt_cols = [
            "Transmembrane pressure, bar",
            "NaCl concentration, ppm",
            "MgSO4 concentration, ppm",
            "Na2SO4 concentration, ppm",
            "MgCl2 concentration, ppm",
            "CaCl2 concentration, ppm",
        ]
        for i, col in enumerate(salt_cols):
            with cols[i % 2]:
                values[col] = numeric_input(col, infer_default_numeric(df, col, bundles))

    submitted = st.button("Predict all targets", type="primary")
    return submitted, values


def clean_selected_values_for_model(selected_values: Dict[str, object]) -> Dict[str, object]:
    cleaned = {}
    for k, v in selected_values.items():
        cleaned[k] = display_to_model_value(v)

    if is_none_selected(selected_values.get("Monomer A2 type")):
        cleaned["A1/A2 ratio"] = 0.0
    if is_none_selected(selected_values.get("Additive X1 type in aqueous phase")):
        cleaned["Additive X1 concentration, wt%"] = 0.0
    if is_none_selected(selected_values.get("Additive X2 type in aqueous phase")):
        cleaned["Additive X2 concentration, wt%"] = 0.0
    if is_none_selected(selected_values.get("Additive Y type in organic phase")):
        cleaned["Additive Y concentration, wt%"] = 0.0
    if is_none_selected(selected_values.get("Nanomaterials type in aqueous phase")):
        cleaned["Nanomaterials loading in aqueous phase, wt%"] = 0.0
    if is_none_selected(selected_values.get("Nanomaterials type in organic phase")):
        cleaned["Nanomaterials loading in organic phase, wt%"] = 0.0

    return cleaned


def make_model_input_from_selected_values(selected_values: Dict[str, object], bundle: Dict) -> pd.DataFrame:
    row: Dict[str, object] = {}
    features = bundle["train_feature_cols"]
    num_cols = bundle.get("num_cols", [])
    cat_cols = bundle.get("cat_cols", [])
    num_fill_map = bundle.get("num_fill_map", {})
    cat_fill_map = bundle.get("cat_fill_map", {})

    cleaned_values = clean_selected_values_for_model(selected_values)

    for feature in features:
        if feature in cleaned_values:
            row[feature] = cleaned_values[feature]
        elif feature in num_cols:
            row[feature] = num_fill_map.get(feature, 0.0)
        elif feature in cat_cols:
            row[feature] = cat_fill_map.get(feature, CATEGORICAL_NONE_INTERNAL_VALUE)
        else:
            row[feature] = num_fill_map.get(feature, 0.0)

    X = pd.DataFrame([row])[features]

    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(num_fill_map.get(col, 0.0))

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("object").fillna(cat_fill_map.get(col, CATEGORICAL_NONE_INTERNAL_VALUE))

    return X


def predict_single_target(selected_values: Dict[str, object], bundle: Dict) -> float:
    X = make_model_input_from_selected_values(selected_values, bundle)

    X_encoded = bundle["encoder"].transform(X)

    for col in bundle["encoded_cols"]:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[bundle["encoded_cols"]]
    X_imputed = bundle["imputer"].transform(X_encoded)

    pred = float(bundle["model"].predict(X_imputed)[0])
    return pred


def predict_all_targets(selected_values: Dict[str, object], bundles: Dict[str, dict]) -> pd.DataFrame:
    rows = []

    for target in TARGET_DISPLAY_ORDER:
        bundle = bundles.get(target)
        if bundle is None:
            rows.append({
                "Target": target,
                "Prediction": None,
                "Unit": "%" if "rejection" in target.lower() else "LMH/bar",
                "Status": "Model not found",
            })
            continue

        try:
            pred = predict_single_target(selected_values, bundle)
            rows.append({
                "Target": target,
                "Prediction": round(pred, 2),
                "Unit": "%" if "rejection" in target.lower() else "LMH/bar",
                "Status": "OK",
            })
        except Exception as e:
            rows.append({
                "Target": target,
                "Prediction": None,
                "Unit": "%" if "rejection" in target.lower() else "LMH/bar",
                "Status": f"Prediction error: {e}",
            })

    return pd.DataFrame(rows)


def model_status_table(bundles: Dict[str, dict], missing_files: List[str]) -> pd.DataFrame:
    rows = []

    for target in TARGET_DISPLAY_ORDER:
        file_name = TARGET_BUNDLE_FILES[target]
        model_path = MODEL_DIR / file_name
        rows.append({
            "Target": label_text(target),
            "Bundle": "OK" if target in bundles else "Missing",
            "Path": str(model_path),
        })

    if missing_files:
        rows.append({
            "Target": "Missing files",
            "Bundle": "Check path",
            "Path": "; ".join(missing_files),
        })

    return pd.DataFrame(rows)


def show_prediction_cards(result_df: pd.DataFrame) -> None:
    ok_df = result_df[result_df["Status"] == "OK"].copy()
    if ok_df.empty:
        display_df = result_df.copy()
        display_df["Target"] = display_df["Target"].map(label_text)
        st.dataframe(display_df, use_container_width=True)
        return

    cols = st.columns(3)
    for i, row in ok_df.reset_index(drop=True).iterrows():
        with cols[i % 3]:
            value = row["Prediction"]
            unit = row["Unit"]
            st.metric(label_text(row["Target"]), f"{value:.2f} {unit}")

    display_df = result_df.copy()
    display_df["Target"] = display_df["Target"].map(label_text)
    st.dataframe(display_df, use_container_width=True)


def result_value(result_df: pd.DataFrame, target: str) -> float | None:
    row = result_df[result_df["Target"] == target]
    if row.empty:
        return None
    val = row["Prediction"].iloc[0]
    if pd.isna(val):
        return None
    return float(val)


def show_radar_chart(result_df: pd.DataFrame, df_base: pd.DataFrame) -> None:
    rejection_targets = [t for t in TARGET_DISPLAY_ORDER if "rejection" in t.lower()]
    labels = [short_target_label(t) for t in rejection_targets] + ["Permeability"]

    rejection_values = [result_value(result_df, t) or 0.0 for t in rejection_targets]
    perm = result_value(result_df, "Permeability, LMH/bar") or 0.0

    if not df_base.empty and "Permeability, LMH/bar" in df_base.columns:
        perm_series = pd.to_numeric(df_base["Permeability, LMH/bar"], errors="coerce").dropna()
        perm_ref = float(perm_series.quantile(0.95)) if len(perm_series) else max(perm, 1.0)
    else:
        perm_ref = max(perm, 1.0)

    perm_score = max(0.0, min(100.0, perm / max(perm_ref, 1e-9) * 100.0))
    values = rejection_values + [perm_score]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Selected prediction",
            line=dict(color="#ff4b4b", width=3),
            marker=dict(color="#ff4b4b", size=7),
        )
    )
    fig.update_layout(
        title="Radar chart: predicted performance profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Radar chart에서 Permeability는 데이터베이스 95 percentile 기준으로 0–100 정규화하여 표시됨.")


def make_tradeoff_dataframe(df: pd.DataFrame, result_df: pd.DataFrame, salt_target: str) -> pd.DataFrame:
    x_col = "Permeability, LMH/bar"
    if df.empty or x_col not in df.columns or salt_target not in df.columns:
        pred_perm = result_value(result_df, x_col)
        pred_rej = result_value(result_df, salt_target)
        if pred_perm is None or pred_rej is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "Permeability": [pred_perm],
            "Rejection": [pred_rej],
            "Source": ["Selected prediction"],
        })

    base = df[[x_col, salt_target]].copy()
    base[x_col] = pd.to_numeric(base[x_col], errors="coerce")
    base[salt_target] = pd.to_numeric(base[salt_target], errors="coerce")
    base = base.dropna(subset=[x_col, salt_target])
    base = base.rename(columns={x_col: "Permeability", salt_target: "Rejection"})
    base["Source"] = "Database"

    pred_perm = result_value(result_df, x_col)
    pred_rej = result_value(result_df, salt_target)
    if pred_perm is not None and pred_rej is not None:
        selected = pd.DataFrame({
            "Permeability": [pred_perm],
            "Rejection": [pred_rej],
            "Source": ["Selected prediction"],
        })
        base = pd.concat([base, selected], ignore_index=True)

    return base


def show_tradeoff_plots(df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    salt_targets = [t for t in TARGET_DISPLAY_ORDER if "rejection" in t.lower()]
    st.subheader("Trade-off analysis")
    st.caption("Database conditions are shown as blue points; the selected AI-predicted condition is shown as a red point.")

    for i in range(0, len(salt_targets), 2):
        cols = st.columns(2)
        for j, salt_target in enumerate(salt_targets[i:i + 2]):
            with cols[j]:
                plot_df = make_tradeoff_dataframe(df, result_df, salt_target)
                if plot_df.empty:
                    st.info(f"{label_text(salt_target)} trade-off plot 생성 불가: 필요한 컬럼 없음.")
                    continue

                base_df = plot_df[plot_df["Source"] == "Database"]
                sel_df = plot_df[plot_df["Source"] == "Selected prediction"]

                x_max = float(plot_df["Permeability"].max()) if plot_df["Permeability"].notna().any() else 1.0
                y_max = float(plot_df["Rejection"].max()) if plot_df["Rejection"].notna().any() else 100.0
                x_domain = [0, max(x_max * 1.08, 1.0)]
                y_domain = [0, min(max(y_max * 1.08, 100.0), 110.0)]

                database_points = (
                    alt.Chart(base_df)
                    .mark_circle(size=45, opacity=0.45, color="#4C9BE8")
                    .encode(
                        x=alt.X("Permeability:Q", title="Permeability, LMH/bar", scale=alt.Scale(domain=x_domain)),
                        y=alt.Y("Rejection:Q", title=label_text(salt_target), scale=alt.Scale(domain=y_domain)),
                        tooltip=[
                            alt.Tooltip("Permeability:Q", format=".2f"),
                            alt.Tooltip("Rejection:Q", format=".2f"),
                        ],
                    )
                )

                selected_point = (
                    alt.Chart(sel_df)
                    .mark_circle(size=230, opacity=1.0, color="#FF3B30", stroke="white", strokeWidth=1.5)
                    .encode(
                        x="Permeability:Q",
                        y="Rejection:Q",
                        tooltip=[
                            alt.Tooltip("Permeability:Q", format=".2f"),
                            alt.Tooltip("Rejection:Q", format=".2f"),
                            "Source:N",
                        ],
                    )
                )

                chart = (database_points + selected_point).properties(
                    height=330,
                    title=f"Permeability vs {short_target_label(salt_target)} rejection",
                ).interactive()

                st.altair_chart(chart, use_container_width=True)


def actual_vs_predicted_path(target: str) -> Path:
    safe = (
        target.replace("/", "_")
        .replace(",", "")
        .replace(" ", "_")
        .replace("%", "pct")
    )
    candidates = [
        OUTPUT_DIR / f"{safe}_actual_vs_predicted.csv",
        OUTPUT_DIR / f"{safe}_prediction.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_actual_vs_predicted(target: str) -> pd.DataFrame:
    path = actual_vs_predicted_path(target)
    if not path.exists():
        return pd.DataFrame()

    df_plot = pd.read_csv(path)
    for col in ["Actual", "Predicted", "Residual"]:
        if col in df_plot.columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

    required = {"Actual", "Predicted"}
    if not required.issubset(df_plot.columns):
        return pd.DataFrame()

    return df_plot.dropna(subset=["Actual", "Predicted"])


def show_actual_vs_predicted_plot(target: str) -> None:
    df_plot = load_actual_vs_predicted(target)
    if df_plot.empty:
        st.info("Actual vs Predicted CSV가 없음. 모델 학습 결과 CSV를 outputs 폴더에 추가하면 표시됨.")
        return

    min_val = float(min(df_plot["Actual"].min(), df_plot["Predicted"].min()))
    max_val = float(max(df_plot["Actual"].max(), df_plot["Predicted"].max()))
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    min_val -= padding
    max_val += padding

    if "Split" not in df_plot.columns:
        df_plot["Split"] = "Data"

    df_plot = df_plot.copy()
    if "Target" in df_plot.columns:
        df_plot["Target"] = df_plot["Target"].map(label_text)

    line_df = pd.DataFrame({"Actual": [min_val, max_val], "Predicted": [min_val, max_val]})

    tooltip = ["Split", alt.Tooltip("Actual:Q", format=".2f"), alt.Tooltip("Predicted:Q", format=".2f")]
    if "Residual" in df_plot.columns:
        tooltip.append(alt.Tooltip("Residual:Q", format=".2f"))

    scatter = (
        alt.Chart(df_plot)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X("Actual:Q", scale=alt.Scale(domain=[min_val, max_val]), title="Actual"),
            y=alt.Y("Predicted:Q", scale=alt.Scale(domain=[min_val, max_val]), title="Predicted"),
            color=alt.Color("Split:N", title="Split"),
            tooltip=tooltip,
        )
    )
    identity = alt.Chart(line_df).mark_line(strokeDash=[5, 5]).encode(x="Actual:Q", y="Predicted:Q")
    chart = (scatter + identity).properties(height=420, title=f"Actual vs Predicted: {label_text(target)}").interactive()
    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "Download actual vs predicted CSV",
        df_plot.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{target}_actual_vs_predicted.csv",
        mime="text/csv",
    )



def show_reference_caip_section() -> None:
    st.divider()
    st.header("Reference CAIP/IP membrane experiment")
    st.caption("Experimental fabrication condition and measured membrane performance used for comparison.")

    ref_img_candidates = [
        APP_DIR / "reference" / "fabrication.png",
        ROOT_DIR / "reference" / "fabrication.png",
    ]

    ref_img = None
    for p in ref_img_candidates:
        if p.exists():
            ref_img = p
            break

    if ref_img is not None:
        st.image(
            str(ref_img),
            caption="IP / CAIP membrane fabrication process",
            use_column_width=True,
        )
    else:
        st.info("fabrication.png가 없음. GitHub에 webapp/reference/fabrication.png 경로로 업로드하면 그림이 표시됨.")

    st.subheader("SDS loading condition")

    sds_df = pd.DataFrame({
        "Membrane": ["IP", "CAIP-0", "CAIP-0.1", "CAIP-0.3", "CAIP-0.5"],
        "SDS loading concentration (wt%)": [0.0, 0.0, 0.1, 0.3, 0.5],
    })

    st.dataframe(sds_df, use_container_width=True, hide_index=True)

    st.subheader("Experimental rejection performance")

    rejection_df = pd.DataFrame({
        "Membrane": ["IP", "CAIP-0", "CAIP-0.1", "CAIP-0.3", "CAIP-0.5"],
        "NaCl rejection (%)": ["97.2 ± 0.1", "84.1 ± 5.2", "88.6 ± 6.8", "88.1 ± 3.9", "71.7 ± 2.2"],
        "MgCl₂ rejection (%)": ["97.2 ± 0.9", "89.4 ± 5.4", "95.4 ± 1.7", "88.7 ± 7.9", "80.5 ± 7.3"],
        "Na₂SO₄ rejection (%)": ["97.7 ± 1.8", "89.6 ± 4.0", "95.2 ± 2.1", "95.4 ± 2.8", "80.7 ± 0.9"],
    })

    st.dataframe(rejection_df, use_container_width=True, hide_index=True)

    st.subheader("Experimental permeability")

    permeability_df = pd.DataFrame({
        "Membrane": ["IP", "CAIP-0", "CAIP-0.1", "CAIP-0.3", "CAIP-0.5"],
        "NaCl permeability (LMH/bar)": ["1.2 ± 0.3", "2.6 ± 0.3", "4.2 ± 0.1", "4.6 ± 0.2", "4.2 ± 0.1"],
        "MgCl₂ permeability (LMH/bar)": ["1.1 ± 0.1", "2.2 ± 0.1", "3.9 ± 0.1", "4.8 ± 0.1", "4.3 ± 0.3"],
        "Na₂SO₄ permeability (LMH/bar)": ["1.0 ± 0.3", "2.4 ± 1.3", "4.0 ± 0.3", "5.1 ± 0.0", "4.6 ± 0.4"],
    })

    st.dataframe(permeability_df, use_container_width=True, hide_index=True)

    with st.expander("Reference condition summary", expanded=False):
        st.markdown(
            """
            - PSf substrate-based PA TFC RO membrane fabrication
            - MPD aqueous impregnation for 3 min
            - TMC organic solution impregnation for 5 min
            - Heating and washing at 60 °C for 2 min
            - CAIP series based on SDS loading concentration: 0, 0.1, 0.3, and 0.5 wt%
            """
        )


def main() -> None:
    bundles, missing_files = load_bundles()
    df = load_dataset(DATA_PATH)

    with st.sidebar:
        st.header("Mode")
        mode = st.radio("View mode", ["User mode", "Developer mode"], index=0)

        if mode == "Developer mode":
            st.header("Path")
            st.caption(f"Model folder: {MODEL_DIR}")
            st.caption(f"Output folder: {OUTPUT_DIR}")
            st.caption(f"Data path: {DATA_PATH if DATA_PATH else 'Not found'}")
            if st.button("Reload app"):
                st.experimental_rerun()
        else:
            if missing_files:
                st.error("Some model bundles are missing.")
            else:
                st.success("Model interface ready")

    if missing_files:
        st.error("일부 XGB bundle 모델 파일을 찾을 수 없음.")
        st.dataframe(pd.DataFrame({"Missing file": missing_files}), use_container_width=True)
        st.stop()

    if mode == "Developer mode":
        with st.expander("Dataset / model status", expanded=False):
            st.write("Dataset shape:", df.shape)
            st.dataframe(model_status_table(bundles, missing_files), use_container_width=True)

        with st.expander("Actual vs Predicted plot", expanded=True):
            selected_target_for_plot = st.selectbox(
                "Select target",
                TARGET_DISPLAY_ORDER,
                index=0,
                key="actual_pred_target_selector",
                format_func=label_text,
            )
            show_actual_vs_predicted_plot(selected_target_for_plot)

    submitted, selected_values = build_selected_input_panel(df, bundles)

    if submitted:
        result_df = predict_all_targets(selected_values, bundles)

        st.subheader("Prediction result")
        show_prediction_cards(result_df)

        st.divider()
        show_radar_chart(result_df, df)
        show_tradeoff_plots(df, result_df)
        show_reference_caip_section()

        st.download_button(
            "Download prediction result CSV",
            result_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="membrane_prediction_result.csv",
            mime="text/csv",
        )

        with st.expander("Selected input values", expanded=False):
            display_values = pd.DataFrame([selected_values])
            st.dataframe(display_values, use_container_width=True)


if __name__ == "__main__":
    main()
