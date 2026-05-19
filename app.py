import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = r"""# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(page_title="Mini Project B Forecasting Starter", layout="wide")

st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption("This starter prepares the dataset, baseline features, export files, and AI grading evidence. Students add models, metrics, and extra dashboard visuals.")


def get_api_key():
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
        if key:
            return key
    except Exception:
        pass

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key

    return st.text_input("OpenRouter API key", type="password", help="Used only when you click the AI grader button.")


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


def audit_dataframe(data):
    audit = pd.DataFrame({
        "column": data.columns,
        "dtype": [str(data[c].dtype) for c in data.columns],
        "missing_percent": [float(data[c].isna().mean() * 100) for c in data.columns],
        "unique_count": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })
    return audit


def clean_time_series(data, timestamp_col, target_col):
    work = data.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)
    return work


def prepare_series(data, timestamp_col, target_col, resample_rule):
    work = clean_time_series(data, timestamp_col, target_col)

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    grouped = work[[timestamp_col] + numeric_cols].groupby(timestamp_col, as_index=False).mean(numeric_only=True)
    grouped = grouped.sort_values(timestamp_col).set_index(timestamp_col)

    if resample_rule != "None":
        grouped = grouped.resample(resample_rule).mean(numeric_only=True)
        grouped[target_col] = grouped[target_col].interpolate(limit_direction="both")

    grouped = grouped.reset_index()
    return grouped


def make_baseline_features(data, timestamp_col, target_col, horizon):
    feat = data[[timestamp_col, target_col]].copy()
    feat = feat.sort_values(timestamp_col)
    feat["lag_1"] = feat[target_col].shift(1)
    feat["lag_24"] = feat[target_col].shift(24)
    feat["rolling_mean_24"] = feat[target_col].shift(1).rolling(24).mean()
    feat["hour"] = feat[timestamp_col].dt.hour
    feat["weekend"] = feat[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    feat["month"] = feat[timestamp_col].dt.month
    feat["y_target"] = feat[target_col].shift(-int(horizon))

    feature_cols = ["lag_1", "lag_24", "rolling_mean_24", "hour", "weekend", "month"]
    feature_table = feat.dropna(subset=feature_cols + ["y_target"]).copy()
    X = feature_table[feature_cols]
    y = feature_table["y_target"]
    return feature_table, X, y, feature_cols


def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_project_card(payload):
    lines = [
        "# Project Card — Mini Project B",
        "",
        f"Student name: {payload['student_name']}",
        f"Student ID: {payload['student_id']}",
        f"Project title: {payload['project_title']}",
        f"Project goal: {payload['project_goal']}",
        f"Dataset rows after cleaning: {payload['dataset']['rows_after_cleaning']}",
        f"Timestamp column: {payload['dataset']['timestamp_column']}",
        f"Target column: {payload['dataset']['target_column']}",
        f"Resampling: {payload['dataset']['resampling']}",
        f"Forecast horizon: {payload['forecast_horizon']}",
        "",
        "## Evidence flags",
        f"- Has metrics table: {payload['evidence_flags']['has_metrics_table']}",
        f"- Has student modeling additions: {payload['evidence_flags']['has_student_modeling_additions']}",
        f"- Has dashboard additions: {payload['evidence_flags']['has_dashboard_additions']}",
        "",
        "## Student notes",
        payload.get("student_notes", ""),
    ]
    return "\n".join(lines)


with st.sidebar:
    st.header("Student info")
    student_name = st.text_input("Student name", value="Marwa")
    student_id = st.text_input("Student ID", value="PG112S25155")
    deployed_url = st.text_input("Deployed Streamlit URL", value="")
    repo_url = st.text_input("GitHub repo URL", value="")
    project_title = st.text_input("Project title", value="Solar AC Power Forecasting")
    project_goal = st.text_area("Project goal", value="Forecast future AC power from the solar generation time series.")
    student_notes = st.text_area("Student notes / insights", value="")

st.header("1) Load dataset")
default_path = "data/dataset_sample.csv"
data_path = st.text_input("Dataset path", value=default_path)

try:
    df = load_data(data_path)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.subheader("First 10 rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset audit")
audit = audit_dataframe(df)
st.dataframe(audit, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.write("Missing percentage — top 10")
    st.dataframe(audit.sort_values("missing_percent", ascending=False).head(10), use_container_width=True)
with col2:
    st.write("Dataset shape")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")

st.header("2) Select time-series columns")
timestamp_options = df.columns.tolist()
numeric_guess = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]

timestamp_col = st.selectbox(
    "Timestamp column",
    timestamp_options,
    index=timestamp_options.index("DATE_TIME") if "DATE_TIME" in timestamp_options else 0,
)
target_col = st.selectbox(
    "Target column",
    numeric_guess if numeric_guess else timestamp_options,
    index=(numeric_guess.index("AC_POWER") if "AC_POWER" in numeric_guess else 0) if numeric_guess else 0,
)

resample_rule = st.selectbox(
    "Optional resampling",
    ["None", "15min", "30min", "1H", "1D"],
    index=0,
)
forecast_horizon = st.number_input("Forecast horizon in rows after optional resampling", min_value=1, max_value=168, value=24, step=1)

ts_data = prepare_series(df, timestamp_col, target_col, resample_rule)
feature_table, X, y, feature_cols = make_baseline_features(ts_data, timestamp_col, target_col, int(forecast_horizon))

st.subheader("Cleaned time-series summary")
summary_cols = st.columns(4)
summary_cols[0].metric("Clean rows", f"{len(ts_data):,}")
summary_cols[1].metric("Feature rows", f"{len(feature_table):,}")
summary_cols[2].metric("Start", str(ts_data[timestamp_col].min()))
summary_cols[3].metric("End", str(ts_data[timestamp_col].max()))

st.line_chart(ts_data.set_index(timestamp_col)[target_col])

st.header("3) Baseline feature table")
st.write("The starter creates baseline features only. Students must add models, metrics, and extra dashboard elements below.")
st.dataframe(feature_table.head(20), use_container_width=True)
st.write("Feature columns:", feature_cols)
st.write("X shape:", X.shape, "y shape:", y.shape)

st.header("4) STUDENT ADDITIONS — MODELING")
st.info("Paste your modeling code under this marker. Create a metrics table named results_df.")
st.code("""
# STUDENT ADDITIONS - MODELING
# Add a time-based split, forecasting models, predictions, and metrics here.
# Required student output:
# results_df = a pandas DataFrame containing model names and metrics.
results_df = None
""", language="python")

results_df = None

st.header("5) STUDENT ADDITIONS — DASHBOARD")
st.info("Paste extra plots, KPIs, and insights under this marker.")
st.code("""
# STUDENT ADDITIONS - DASHBOARD
# Add at least one extra plot or KPI here.
# Explain key insights in the student notes box.
""", language="python")

st.header("6) Export submission files")

has_metrics_table = isinstance(results_df, pd.DataFrame)
results_table = [] if results_df is None else results_df.to_dict(orient="records")

submission = {
    "student_name": student_name,
    "student_id": student_id,
    "deployed_url": deployed_url,
    "repo_url": repo_url,
    "project_title": project_title,
    "project_goal": project_goal,
    "student_notes": student_notes,
    "dataset": {
        "path": data_path,
        "rows_raw": int(len(df)),
        "rows_after_cleaning": int(len(ts_data)),
        "feature_rows": int(len(feature_table)),
        "timestamp_column": timestamp_col,
        "target_column": target_col,
        "start_time": str(ts_data[timestamp_col].min()),
        "end_time": str(ts_data[timestamp_col].max()),
        "resampling": resample_rule,
        "missing_percent_top10": audit.sort_values("missing_percent", ascending=False).head(10).to_dict(orient="records"),
    },
    "forecast_horizon": int(forecast_horizon),
    "baseline_features": feature_cols,
    "evidence_flags": {
        "has_metrics_table": bool(has_metrics_table),
        "has_student_modeling_additions": bool(has_metrics_table),
        "has_dashboard_additions": bool(student_notes.strip()),
        "discusses_missing_timestamps_outliers_resampling": bool(student_notes.strip()),
    },
    "results_table": results_table,
}

submission_json = json.dumps(submission, indent=2)
project_card_md = build_project_card(submission)

st.download_button(
    "Download submission.json",
    data=submission_json,
    file_name="submission.json",
    mime="application/json",
)

st.download_button(
    "Download project_card.md",
    data=project_card_md,
    file_name="project_card.md",
    mime="text/markdown",
)

st.header("7) AI grader out of 80")
st.warning("The AI grader is strict and uses only the evidence in submission.json. Add your own models, metrics, dashboard, and insights before final grading.")
api_key = get_api_key()

if st.button("Run AI grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key.")
    else:
        grading_prompt = AI_GRADER_PROMPT_TEMPLATE.replace("<insert submission.json contents here>", submission_json)
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "user", "content": grading_prompt},
                    ],
                    "temperature": 0,
                },
                timeout=60,
            )
            response.raise_for_status()
            raw_output = response.json()["choices"][0]["message"]["content"]
            try:
                parsed = safe_json_loads(raw_output)
                st.json(parsed)
            except Exception:
                st.subheader("Raw AI output")
                st.code(raw_output, language="json")
        except Exception as exc:
            st.error(f"AI grader failed: {exc}")
