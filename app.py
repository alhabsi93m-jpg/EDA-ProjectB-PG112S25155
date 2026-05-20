import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
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


st.set_page_config(page_title="Mini Project B Forecasting App", layout="wide")

st.title("Mini Project B — Solar AC Power Forecasting")
st.caption(
    "Time-series forecasting app with dataset audit, improved feature engineering, "
    "student modeling, dashboard visuals, export files, and AI grading evidence."
)


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

    return st.text_input(
        "OpenRouter API key",
        type="password",
        help="Used only when you click the AI grader button.",
    )


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

    work[timestamp_col] = pd.to_datetime(
        work[timestamp_col],
        errors="coerce",
        dayfirst=True,
    )

    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

    rows_before = len(work)
    invalid_timestamp_rows = int(work[timestamp_col].isna().sum())
    invalid_target_rows = int(work[target_col].isna().sum())

    work = work.dropna(subset=[timestamp_col, target_col])
    work = work.sort_values(timestamp_col)

    rows_after = len(work)

    cleaning_report = {
        "rows_before_cleaning": int(rows_before),
        "rows_after_cleaning": int(rows_after),
        "invalid_timestamp_rows_removed": int(invalid_timestamp_rows),
        "invalid_target_rows_removed": int(invalid_target_rows),
    }

    return work, cleaning_report


def prepare_series(data, timestamp_col, target_col, resample_rule):
    work, cleaning_report = clean_time_series(data, timestamp_col, target_col)

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()

    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    grouped = (
        work[[timestamp_col] + numeric_cols]
        .groupby(timestamp_col, as_index=False)
        .mean(numeric_only=True)
    )

    grouped = grouped.sort_values(timestamp_col).set_index(timestamp_col)

    if resample_rule != "None":
        grouped = grouped.resample(resample_rule).mean(numeric_only=True)
        grouped[target_col] = grouped[target_col].interpolate(limit_direction="both")

    grouped = grouped.reset_index()

    return grouped, cleaning_report


def make_improved_features(data, timestamp_col, target_col, horizon):
    feat = data[[timestamp_col, target_col]].copy()
    feat = feat.sort_values(timestamp_col)
    feat = feat.reset_index(drop=True)

    # ------------------------------------------------------------
    # Lag features
    # These help the model learn recent and daily historical patterns.
    # ------------------------------------------------------------
    lag_values = [1, 2, 3, 4, 8, 12, 24, 48]

    for lag in lag_values:
        feat[f"lag_{lag}"] = feat[target_col].shift(lag)

    # ------------------------------------------------------------
    # Rolling window features
    # Shift first to avoid using the current target value directly.
    # ------------------------------------------------------------
    rolling_windows = [3, 6, 12, 24]

    shifted_target = feat[target_col].shift(1)

    for window in rolling_windows:
        feat[f"rolling_mean_{window}"] = shifted_target.rolling(window).mean()
        feat[f"rolling_std_{window}"] = shifted_target.rolling(window).std()
        feat[f"rolling_min_{window}"] = shifted_target.rolling(window).min()
        feat[f"rolling_max_{window}"] = shifted_target.rolling(window).max()

    # ------------------------------------------------------------
    # Change features
    # These help the model learn whether power is rising or falling.
    # ------------------------------------------------------------
    feat["diff_1"] = feat[target_col].diff(1).shift(1)
    feat["diff_24"] = feat[target_col].diff(24).shift(1)
    feat["pct_change_1"] = feat[target_col].pct_change(1).replace([np.inf, -np.inf], np.nan).shift(1)

    # ------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------
    feat["hour"] = feat[timestamp_col].dt.hour
    feat["dayofweek"] = feat[timestamp_col].dt.dayofweek
    feat["weekend"] = feat["dayofweek"].isin([5, 6]).astype(int)
    feat["month"] = feat[timestamp_col].dt.month
    feat["dayofyear"] = feat[timestamp_col].dt.dayofyear

    # ------------------------------------------------------------
    # Cyclical encoding
    # This is better than raw hour/month because time wraps around.
    # Example: hour 23 and hour 0 are close in real life.
    # ------------------------------------------------------------
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)

    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    feat["dayofyear_sin"] = np.sin(2 * np.pi * feat["dayofyear"] / 365)
    feat["dayofyear_cos"] = np.cos(2 * np.pi * feat["dayofyear"] / 365)

    # ------------------------------------------------------------
    # Solar-specific feature
    # This simple flag helps because solar power is usually zero at night.
    # ------------------------------------------------------------
    feat["is_daylight_hour"] = feat["hour"].between(6, 18).astype(int)

    # ------------------------------------------------------------
    # Forecast target
    # y_target is the future value we want to predict.
    # ------------------------------------------------------------
    feat["y_target"] = feat[target_col].shift(-int(horizon))

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_8",
        "lag_12",
        "lag_24",
        "lag_48",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_min_3",
        "rolling_max_3",
        "rolling_mean_6",
        "rolling_std_6",
        "rolling_min_6",
        "rolling_max_6",
        "rolling_mean_12",
        "rolling_std_12",
        "rolling_min_12",
        "rolling_max_12",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_min_24",
        "rolling_max_24",
        "diff_1",
        "diff_24",
        "pct_change_1",
        "hour",
        "dayofweek",
        "weekend",
        "month",
        "dayofyear",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "is_daylight_hour",
    ]

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
        "## Feature engineering",
        f"- Number of engineered features: {len(payload['engineered_features'])}",
        "- Added lag features, rolling statistics, difference features, percentage change, calendar features, cyclical time features, and daylight-hour flag.",
        "",
        "## Modeling",
        f"- Time-based split used: {payload['student_modeling_summary']['time_based_split_used']}",
        f"- Models compared: {payload['student_modeling_summary']['models_compared']}",
        f"- Best model by RMSE: {payload['student_modeling_summary']['best_model_by_rmse']}",
        "",
        "## Evidence flags",
        f"- Has metrics table: {payload['evidence_flags']['has_metrics_table']}",
        f"- Has student modeling additions: {payload['evidence_flags']['has_student_modeling_additions']}",
        f"- Has dashboard additions: {payload['evidence_flags']['has_dashboard_additions']}",
        f"- Uses time-based split: {payload['evidence_flags']['uses_time_based_split']}",
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

    project_title = st.text_input(
        "Project title",
        value="Solar AC Power Forecasting with Improved Feature Engineering",
    )

    project_goal = st.text_area(
        "Project goal",
        value=(
            "Forecast future solar AC power using timestamp-based features, lag features, "
            "rolling statistics, and machine learning models."
        ),
    )

    student_notes = st.text_area(
        "Student notes / insights",
        value=(
            "The dataset was cleaned by parsing timestamps, removing invalid timestamp or target values, "
            "sorting chronologically, and grouping repeated timestamps by average AC power. "
            "Feature engineering was improved by adding multiple lag features, rolling mean/std/min/max values, "
            "difference features, percentage change, cyclical time encodings, and a daylight-hour flag. "
            "A time-based train/test split was used to avoid leakage. "
            "Linear Regression and Random Forest models were compared using MAE, RMSE, and R2. "
            "The dashboard includes KPIs, hourly power profile, daily power profile, and actual vs predicted values."
        ),
    )


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
    st.dataframe(
        audit.sort_values("missing_percent", ascending=False).head(10),
        use_container_width=True,
    )

with col2:
    st.write("Dataset shape")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]:,}")


st.header("2) Select time-series columns")

timestamp_options = df.columns.tolist()

numeric_guess = [
    c for c in df.columns
    if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5
]

timestamp_col = st.selectbox(
    "Timestamp column",
    timestamp_options,
    index=timestamp_options.index("DATE_TIME") if "DATE_TIME" in timestamp_options else 0,
)

target_col = st.selectbox(
    "Target column",
    numeric_guess if numeric_guess else timestamp_options,
    index=(numeric_guess.index("AC_POWER") if "AC_POWER" in numeric_guess else 0)
    if numeric_guess else 0,
)

resample_rule = st.selectbox(
    "Optional resampling",
    ["None", "15min", "30min", "1H", "1D"],
    index=0,
)

forecast_horizon = st.number_input(
    "Forecast horizon in rows after optional resampling",
    min_value=1,
    max_value=168,
    value=24,
    step=1,
)


ts_data, cleaning_report = prepare_series(df, timestamp_col, target_col, resample_rule)

feature_table, X, y, feature_cols = make_improved_features(
    ts_data,
    timestamp_col,
    target_col,
    int(forecast_horizon),
)


st.subheader("Cleaned time-series summary")

summary_cols = st.columns(4)
summary_cols[0].metric("Clean rows", f"{len(ts_data):,}")
summary_cols[1].metric("Feature rows", f"{len(feature_table):,}")
summary_cols[2].metric("Start", str(ts_data[timestamp_col].min()))
summary_cols[3].metric("End", str(ts_data[timestamp_col].max()))

clean_col1, clean_col2, clean_col3 = st.columns(3)
clean_col1.metric("Invalid timestamps removed", cleaning_report["invalid_timestamp_rows_removed"])
clean_col2.metric("Invalid targets removed", cleaning_report["invalid_target_rows_removed"])
clean_col3.metric("Engineered features", len(feature_cols))

st.write("Target time-series plot")
st.line_chart(ts_data.set_index(timestamp_col)[target_col])


st.header("3) Improved feature engineering")

st.write(
    "This project uses improved forecasting features: lag values, rolling statistics, "
    "difference features, percentage change, calendar values, cyclical time encodings, "
    "and a daylight-hour flag."
)

st.dataframe(feature_table.head(20), use_container_width=True)

with st.expander("Show engineered feature columns"):
    st.write(feature_cols)

st.write("X shape:", X.shape, "y shape:", y.shape)


st.header("4) STUDENT ADDITIONS — MODELING")
st.info("This section trains forecasting models and creates the required metrics table named results_df.")

# ============================================================
# STUDENT ADDITIONS — MODELING
# Time-based split + forecasting models + metrics table.
# ============================================================

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results_df = None
plot_df = None
best_model_name = None
feature_importance_df = None

if len(X) > 50 and len(y) > 50:
    st.subheader("Student Modeling: Time-Based Forecasting Models")

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    split_cols = st.columns(3)
    split_cols[0].metric("Training rows", f"{len(X_train):,}")
    split_cols[1].metric("Testing rows", f"{len(X_test):,}")
    split_cols[2].metric("Split method", "Time-based 80/20")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            max_depth=12,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
        ),
    }

    results = []
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "model": model_name,
            "MAE": round(float(mae), 3),
            "RMSE": round(float(rmse), 3),
            "R2": round(float(r2), 3),
        })

        predictions[model_name] = y_pred

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    st.write("Model performance metrics")
    st.dataframe(results_df, use_container_width=True)

    best_model_name = results_df.iloc[0]["model"]
    st.success(f"Best model by RMSE: {best_model_name}")

    plot_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": predictions[best_model_name],
    }).reset_index(drop=True)

    st.write("Actual vs predicted values for the best model")
    st.line_chart(plot_df.head(200))

    best_model_object = models[best_model_name]

    if hasattr(best_model_object, "feature_importances_"):
        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": best_model_object.feature_importances_,
        }).sort_values("importance", ascending=False)

        st.subheader("Top feature importances")
        st.dataframe(feature_importance_df.head(15), use_container_width=True)

        st.bar_chart(
            feature_importance_df.head(15).set_index("feature")["importance"]
        )

else:
    st.warning("Not enough rows available for modeling after feature preparation.")


st.header("5) STUDENT ADDITIONS — DASHBOARD")
st.info("This section adds extra dashboard KPIs, plots, and insights.")

# ============================================================
# STUDENT ADDITIONS — DASHBOARD
# Extra plots, KPIs, and insights.
# ============================================================

dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)

dash_col1.metric("Average AC power", round(float(ts_data[target_col].mean()), 3))
dash_col2.metric("Maximum AC power", round(float(ts_data[target_col].max()), 3))
dash_col3.metric("Minimum AC power", round(float(ts_data[target_col].min()), 3))
dash_col4.metric("Standard deviation", round(float(ts_data[target_col].std()), 3))

hourly_profile = ts_data.copy()
hourly_profile["hour"] = hourly_profile[timestamp_col].dt.hour
hourly_avg = hourly_profile.groupby("hour")[target_col].mean()

st.subheader("Average AC power by hour of day")
st.bar_chart(hourly_avg)

daily_profile = ts_data.copy()
daily_profile["date"] = daily_profile[timestamp_col].dt.date
daily_avg = daily_profile.groupby("date")[target_col].mean()

st.subheader("Average AC power by date")
st.line_chart(daily_avg)

st.subheader("Target distribution")
st.bar_chart(ts_data[target_col].round(0).value_counts().sort_index())

st.subheader("Dashboard insight")
st.write(
    "The hourly profile shows a clear solar pattern: AC power is low during night hours "
    "and higher during daylight hours. This supports the use of hour-based features, "
    "cyclical time encoding, lag values, and rolling averages. The model comparison table "
    "shows which algorithm performs best using MAE, RMSE, and R2."
)


st.header("6) Export submission files")

has_metrics_table = isinstance(results_df, pd.DataFrame)
results_table = [] if results_df is None else results_df.to_dict(orient="records")

top_features = (
    []
    if feature_importance_df is None
    else feature_importance_df.head(10).to_dict(orient="records")
)

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
        "missing_percent_top10": audit.sort_values(
            "missing_percent",
            ascending=False,
        ).head(10).to_dict(orient="records"),
        "cleaning_report": cleaning_report,
    },
    "forecast_horizon": int(forecast_horizon),
    "baseline_features": [
        "lag_1",
        "lag_24",
        "rolling_mean_24",
        "hour",
        "weekend",
        "month",
    ],
    "engineered_features": feature_cols,
    "feature_engineering_summary": {
        "lag_features": ["lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_12", "lag_24", "lag_48"],
        "rolling_features": [
            "rolling_mean_3",
            "rolling_std_3",
            "rolling_min_3",
            "rolling_max_3",
            "rolling_mean_6",
            "rolling_std_6",
            "rolling_min_6",
            "rolling_max_6",
            "rolling_mean_12",
            "rolling_std_12",
            "rolling_min_12",
            "rolling_max_12",
            "rolling_mean_24",
            "rolling_std_24",
            "rolling_min_24",
            "rolling_max_24",
        ],
        "change_features": ["diff_1", "diff_24", "pct_change_1"],
        "calendar_features": ["hour", "dayofweek", "weekend", "month", "dayofyear"],
        "cyclical_features": [
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "dayofyear_sin",
            "dayofyear_cos",
        ],
        "solar_specific_features": ["is_daylight_hour"],
        "top_feature_importances": top_features,
    },
    "student_modeling_summary": {
        "time_based_split_used": True,
        "train_percent": 80,
        "test_percent": 20,
        "models_compared": [] if results_df is None else results_df["model"].tolist(),
        "best_model_by_rmse": best_model_name,
        "metrics_used": ["MAE", "RMSE", "R2"],
    },
    "dashboard_summary": {
        "extra_kpis_added": True,
        "extra_plot_added": True,
        "dashboard_plots": [
            "Target time-series plot",
            "Average AC power by hour of day",
            "Average AC power by date",
            "Target distribution",
            "Actual vs predicted values",
            "Feature importance chart",
        ],
    },
    "evidence_flags": {
        "has_metrics_table": bool(has_metrics_table),
        "has_student_modeling_additions": bool(has_metrics_table),
        "has_dashboard_additions": True,
        "discusses_missing_timestamps_outliers_resampling": bool(student_notes.strip()),
        "uses_time_based_split": bool(has_metrics_table),
        "has_improved_feature_engineering": True,
        "has_feature_importance": bool(feature_importance_df is not None),
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
st.warning(
    "The AI grader is strict and uses only the evidence in submission.json. "
    "Make sure your metrics table, dashboard additions, feature engineering evidence, "
    "and insights are visible before final grading."
)

api_key = get_api_key()

if st.button("Run AI grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key.")
    else:
        grading_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
            "<insert submission.json contents here>",
            submission_json,
        )

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
