import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from xai_utils import (
    plot_residuals,
    plot_feature_importance,
    plot_shap_summary,
)

from rag_utils import rag_answer  # RAG-based chatbot helper

# -----------------------------
# BASIC CONFIG
# -----------------------------
st.set_page_config(
    page_title="Banff Traffic & Visitors – CMPT 3835",
    layout="wide"
)

TARGET_COL = "daily_visits.1"  # target = tomorrow's visits (scaled)
LEAKAGE_COLS = ["daily_visits"]   # today's visits (very close to tomorrow's)


# -----------------------------
# LOAD DATA & MODEL (with cache)
# -----------------------------
@st.cache_data
def load_data():
    data_path = "data/final_selected_features.csv"
    if not os.path.exists(data_path):
        st.error(f"Could not find data file at {data_path}")
        return None
    df = pd.read_csv(data_path)
    return df


@st.cache_resource
def load_model():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Could not find model file at {model_path}")
        return None
    model = joblib.load(model_path)
    return model


df = load_data()
model = load_model()

if df is None or model is None:
    st.stop()

# Split X and y for evaluation
if TARGET_COL in df.columns:
    y_all = df[TARGET_COL].copy()
    # IMPORTANT: drop target + leakage feature to match the retrained model
    X_all = df.drop(
        columns=[TARGET_COL] + [c for c in LEAKAGE_COLS if c in df.columns]
    ).copy()
else:
    y_all = None
    X_all = df.copy()


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "EDA & Feature Engineering", "Model & Prediction", "XAI", "RAG Chatbot"]
)


# -----------------------------
# HOME PAGE
# -----------------------------
def show_home():
    st.title("Banff Traffic & Visitor Prediction App")
    st.write(
        """
        This app is part of **CMPT-3835 – Banff Traffic Management**.

        It uses your cleaned and engineered dataset to:
        - Explore visitor and traffic patterns (EDA & feature engineering)
        - Show how the ML model is performing
        - Provide a simple prediction demo for daily visitors
        - Show XAI graphs to explain the model
        - Provide a RAG-based chatbot interface for natural language questions

        Data source: `final_selected_features.csv`  
        Best model: XGBoost pipeline (saved as `best_model.pkl`), trained **without**
        the shortcut feature `daily_visits` to avoid leakage.
        """
    )

    st.subheader("Quick Dataset Preview")
    st.write("First 5 rows of the final features table:")
    st.dataframe(df.head())

    st.write("Shape of the dataset:")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# -----------------------------
# EDA & FEATURE ENGINEERING PAGE
# -----------------------------
def show_eda():
    st.title("EDA & Feature Engineering")

    st.write(
        """
        This page shows the main EDA and feature engineering graphs we used
        to understand Banff parking demand and visitor behaviour.
        """
    )

    # 1) Site / parking location counts
    st.markdown("### 1. Parking Site Usage Distribution")
    st.image("Picture1.png", use_container_width=True)
    st.markdown(
        """
- Shows how parking records are distributed across different sites in Banff  
- Taller bars mean that location is used more often and has higher parking demand  
- Helps identify which streets or areas face the most parking pressure
"""
    )

    # 2) Payment method share
    st.markdown("### 2. Payment Method Share")
    st.image("Picture2.png", use_container_width=True)
    st.markdown(
        """
- Displays the share of each payment type such as bank card, mobile, and cash  
- Digital methods (card / phone) dominate, while cash usage is very small  
- Suggests that most visitors are comfortable with contactless and app-based payments
"""
    )

    # 3) Parking usage by hour of day
    st.markdown("### 3. Parking Usage by Hour of the Day")
    st.image("Picture3.png", use_container_width=True)
    st.markdown(
        """
- Shows how parking demand changes over the hours of the day  
- Early morning is quiet, demand rises through late morning and afternoon, then drops in the evening  
- Helps the Town see peak hours when congestion and parking pressure are highest
"""
    )

    # 4) Feature correlation heatmap
    st.markdown("### 4. Feature Correlation with Daily Visitors")
    st.image("Picture4.png", use_container_width=True)
    st.markdown(
        """
- Shows how strongly each engineered feature is correlated with daily visitor counts  
- Warmer colours mean a stronger positive relationship with visits  
- Confirms that lag features, rolling averages, and weekend/holiday flags are useful for prediction
"""
    )


# -----------------------------
# MODEL & PREDICTION PAGE
# -----------------------------
def show_model_and_prediction():
    st.title("Model Performance & Simple Prediction Demo")

    # =========================
    # 1. Overall model performance
    # =========================
    st.subheader("1. Overall model performance on full dataset")

    if y_all is not None:
        # Predict on the full dataset (for a simple evaluation)
        y_pred_all = model.predict(X_all)
        mae = mean_absolute_error(y_all, y_pred_all)
        rmse = math.sqrt(mean_squared_error(y_all, y_pred_all))
        r2 = r2_score(y_all, y_pred_all)
        mape = mean_absolute_percentage_error(y_all, y_pred_all) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE (lower is better)", f"{mae:.4f}")
        col2.metric("RMSE (lower is better)", f"{rmse:.4f}")
        col3.metric("R² Score (closer to 1 is better)", f"{r2:.4f}")
        col4.metric("MAPE (%)", f"{mape:.2f}")

        st.write(
            """
            These metrics show how far, on average, the model's predictions are
            from the actual visitor counts (on the scaled target).
            """
        )

        # Simple error distribution bar chart
        errors = y_all - y_pred_all
        fig_err, ax_err = plt.subplots(figsize=(6, 3))
        ax_err.hist(errors, bins=30)
        ax_err.set_title("Error distribution (actual - predicted)")
        ax_err.set_xlabel("Error")
        ax_err.set_ylabel("Count")
        st.pyplot(fig_err)

    else:
        st.warning("Target column not found, cannot compute performance metrics.")

    # =========================
    # 2. Pick a real row and compare
    # =========================
    st.subheader("2. Try a prediction on a real row from your dataset")

    st.write(
        """
        Use the slider below to pick a row from the dataset.  
        The app will show:
        - The input features for that row
        - The model's predicted visitors
        - The actual visitors (if available)
        """
    )

    # Slider to pick a row index
    row_index = st.slider(
        "Select a row index from the dataset",
        min_value=0,
        max_value=len(X_all) - 1,
        value=0,
        step=1,
    )

    X_row = X_all.iloc[[row_index]]  # keep as DataFrame
    st.write("Selected row (features):")
    st.dataframe(X_row)

    # Make prediction
    y_pred_row = model.predict(X_row)[0]

    if y_all is not None:
        y_true_row = y_all.iloc[row_index]
        st.write(f"**Predicted visitors (scaled):** {y_pred_row:.3f}")
        st.write(f"**Actual visitors (scaled):** {y_true_row:.3f}")

        # Small bar chart: predicted vs actual
        fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
        ax_bar.bar(["Actual", "Predicted"], [y_true_row, y_pred_row])
        ax_bar.set_ylabel("Visitors (scaled)")
        ax_bar.set_title("Actual vs Predicted for selected row")
        st.pyplot(fig_bar)
    else:
        st.write(f"**Predicted visitors (scaled):** {y_pred_row:.3f}")
        st.write("Actual visitors not available (no target column).")

    # =========================
    # 3. Custom prediction playground (date + month + parking + sliders)
    # =========================
    st.subheader("3. Build a custom prediction (What-if scenario)")

    st.write(
        """
        Change the options below to create your own scenario.  
        The model will use these settings to predict how many visitors (scaled)
        might come to Banff for that scenario.
        """
    )

    # --- DATE PICKER: user chooses a day, we derive month + weekend/weekday ---
    today = pd.Timestamp.today().date()
    selected_date = st.date_input("Select a date for this scenario", value=today)
    derived_month = selected_date.month
    is_weekend_from_date = 1 if selected_date.weekday() >= 5 else 0
    st.caption(
        f"Selected date is {selected_date.strftime('%A, %d %b %Y')} "
        f"→ month = {derived_month}, "
        f"{'weekend' if is_weekend_from_date == 1 else 'weekday'}."
    )

    # --- PARKING LOCATION (context only – model is global) ---
    parking_location = st.selectbox(
        "Parking location (context – current model is for overall visitors)",
        [
            "All Banff (overall visitors)",
            "Downtown core parking",
            "Gondola / Upper Hot Springs",
            "Lake Minnewanka / scenic areas",
            "Other / not specified",
        ],
        index=0,
    )
    st.caption(
        "Note: the current model was trained on overall daily visitors, "
        "not per-parking-lot demand. This option is for scenario context."
    )

    # Use the median of the dataset as a safe starting point
    base_values = X_all.median(numeric_only=True)

    # -------------------------
    # Time-related controls
    # -------------------------
    c1, c2, c3 = st.columns(3)
    month = c1.slider(
        "Month used by the model (1 = Jan, 12 = Dec)",
        1,
        12,
        int(derived_month),  # default from selected date
    )
    hour = c2.slider(
        "Hour of day",
        0,
        23,
        int(base_values.get("hour", 12)),
    )
    is_weekend = c3.selectbox(
        "Is weekend? (model input)",
        [0, 1],
        index=is_weekend_from_date,
        format_func=lambda x: "Weekend" if x == 1 else "Weekday",
    )

    # -------------------------
    # Lag features
    # -------------------------
    c4, c5, c6 = st.columns(3)
    lag_7 = c4.slider(
        "Lag 7 (visitors 7 days ago)",
        0.0,
        float(X_all["lag_7"].max()),
        float(base_values.get("lag_7", 0.0)),
    )
    lag_14 = c5.slider(
        "Lag 14 (visitors 14 days ago)",
        0.0,
        float(X_all["lag_14"].max()),
        float(base_values.get("lag_14", 0.0)),
    )
    lag_30 = c6.slider(
        "Lag 30 (visitors 30 days ago)",
        0.0,
        float(X_all["lag_30"].max()),
        float(base_values.get("lag_30", 0.0)),
    )

    rolling_7 = st.slider(
        "Rolling 7-day average visitors",
        0.0,
        float(X_all["rolling_7"].max()),
        float(base_values.get("rolling_7", 0.0)),
    )

    # -------------------------
    # Holiday options (if present)
    # -------------------------
    c7, c8 = st.columns(2)
    if "is_holiday" in X_all.columns:
        is_holiday = c7.selectbox(
            "Is holiday?",
            [0, 1],
            index=int(base_values.get("is_holiday", 0)),
            format_func=lambda x: "Holiday" if x == 1 else "No",
        )
    else:
        is_holiday = None

    if "is_long_weekend" in X_all.columns:
        is_long_weekend = c8.selectbox(
            "Is long weekend?",
            [0, 1],
            index=int(base_values.get("is_long_weekend", 0)),
            format_func=lambda x: "Long weekend" if x == 1 else "No",
        )
    else:
        is_long_weekend = None

    # -------------------------
    # Build final feature row
    # -------------------------
    custom_row = base_values.copy()

    updates = {
        "month": month,
        "hour": hour,
        "is_weekend": is_weekend,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_30": lag_30,
        "rolling_7": rolling_7,
    }
    if is_holiday is not None:
        updates["is_holiday"] = is_holiday
    if is_long_weekend is not None:
        updates["is_long_weekend"] = is_long_weekend

    for col, val in updates.items():
        if col in X_all.columns:
            custom_row[col] = val

    # Ensure column order matches X_all
    X_custom = pd.DataFrame([custom_row[X_all.columns]])

    if st.button("Predict visitors for this scenario"):
        y_custom = model.predict(X_custom)[0]
        st.success(
            f"Predicted visitors (scaled) for your custom scenario: {y_custom:.3f}"
        )


# -----------------------------
# XAI PAGE
# -----------------------------
def show_xai():
    st.title("Explainable AI (XAI) – Banff Visitors Model")

    if y_all is None:
        st.warning("Target column not found, cannot compute XAI plots.")
        return

    st.markdown(
        """
        The goal of this page is to **explain how the model makes decisions**.

        We use:
        - Residual plots to see error patterns  
        - Global feature importance  
        - SHAP values for top features  
        """
    )

    # 1) Residual plot
    st.subheader("1. Residual Plot (Global Error Behaviour)")
    fig_resid = plot_residuals(model, X_all, y_all)
    st.pyplot(fig_resid)

    st.markdown("---")

    # 2) Permutation feature importance (global)
    st.subheader("2. Global Feature Importance (Permutation Importance)")
    fig_perm = plot_feature_importance(model, X_all, y_all, top_n=15)
    st.pyplot(fig_perm)

    st.markdown("---")

    # 3) SHAP summary (global)
    st.subheader("3. Global SHAP Summary (Top Features)")
    try:
        with st.spinner("Computing SHAP explanations on a sample of the data..."):
            fig_shap = plot_shap_summary(model, X_all)
        st.pyplot(fig_shap)
    except Exception:
        st.warning(
            "SHAP summary could not be computed for this saved model. "
            "We will rely on the residual plot and permutation feature "
            "importance as our main global XAI tools."
        )


# -----------------------------
# RAG CHATBOT PAGE
# -----------------------------
def show_rag_chatbot():
    st.title("RAG-based Chatbot – Banff Visitors")

    st.write(
        """
        This chatbot uses a simple Retrieval-Augmented Generation (RAG) idea.
        It looks at the Banff dataset and gives short, data-based answers
        to your questions.
        """
    )

    # Fixed questions (from your instructions)
    fixed_questions = [
        "How are visitor counts related to lag_7 and lag_30 values in this dataset?",
        "How do the 7-day rolling averages compare to actual visitor counts in April?",
        "Which days in April stand out as much busier than their recent 7-day trend?",
        "Do weekends show higher visitor numbers than weekdays in April ?",
    ]

    options = fixed_questions + ["Type my own question"]

    selected_option = st.radio(
        "Select a question or choose 'Type my own question':",
        options,
        index=0,
    )

    custom_q = st.text_input(
        "Or type your own question:",
        "",
        placeholder="Example: How do visitors change on holidays compared to normal days?",
    )

    if st.button("Get answer"):
        # Decide which question to send
        if selected_option != "Type my own question":
            q = selected_option
        else:
            q = custom_q.strip()

        if not q:
            st.warning("Please select a question or type your own question.")
            return

        with st.spinner("Thinking..."):
            raw_answer = rag_answer(q, df)

        # Remove the "Example days from the dataset" section if present
        marker = "Example days from the dataset"
        idx = raw_answer.find(marker)
        if idx != -1:
            answer = raw_answer[:idx].strip()
        else:
            answer = raw_answer.strip()

        st.subheader("Answer")
        st.markdown(answer)


# -----------------------------
# ROUTER
# -----------------------------
if page == "Home":
    show_home()
elif page == "EDA & Feature Engineering":
    show_eda()
elif page == "Model & Prediction":
    show_model_and_prediction()
elif page == "XAI":
    show_xai()
elif page == "RAG Chatbot":
    show_rag_chatbot()
