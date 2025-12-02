import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt

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

TARGET_COL = "daily_visits.1"  # change only if your target is different
# Features that behave like leakage / shortcuts for the target
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
    X_all = df.drop(columns=[TARGET_COL]).copy()
else:
    y_all = None
    X_all = df.copy()


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "EDA", "Model & Prediction", "XAI", "RAG Chatbot"]
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
        - Explore visitor and traffic patterns (EDA)
        - Show how the ML model is performing
        - Provide a simple prediction demo for daily visitors
        - Show XAI graphs to explain the model
        - Provide a RAG-based chatbot interface for natural language questions

        Data source: `final_selected_features.csv`  
        Best model: tuned XGBoost (saved as `best_model.pkl`)
        """
    )

    st.subheader("Quick Dataset Preview")
    st.write("First 5 rows of the final features table:")
    st.dataframe(df.head())

    st.write("Shape of the dataset:")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# -----------------------------
# EDA PAGE
# -----------------------------
def show_eda():
    st.title("Exploratory Data Analysis (EDA)")

    st.write("Below are some simple views of the final engineered dataset.")

    st.subheader("1. Preview of the data")
    st.dataframe(df.head())

    st.subheader("2. Distribution of the target (daily visitors)")
    if TARGET_COL in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df[TARGET_COL].values)
        ax.set_title("Daily Visitors Over Time")
        ax.set_xlabel("Row index (time order)")
        ax.set_ylabel("Number of visitors")
        st.pyplot(fig)

        st.write(
            """
            This line chart shows how the **daily_visits.1** values change across the dataset.  
            Peaks indicate busier days with more visitors, while lower points show quieter days.
            """
        )
    else:
        st.warning(f"Target column '{TARGET_COL}' not found in the dataset.")

    # If day_of_week exists, show average visitors per day of week
    if TARGET_COL in df.columns and "day_of_week" in df.columns:
        st.subheader("3. Average visitors by day of week")
        avg_by_dow = df.groupby("day_of_week")[TARGET_COL].mean().reset_index()
        st.bar_chart(data=avg_by_dow, x="day_of_week", y=TARGET_COL)
        st.write(
            """
            This bar chart shows the **average daily visitors for each day of the week**.  
            It helps identify which days are normally the busiest.
            """
        )
    else:
        st.info("Column 'day_of_week' not found, so day-of-week plot is skipped.")


# -----------------------------
# MODEL & PREDICTION PAGE
# -----------------------------
def show_model_and_prediction():
    st.title("Model Performance & Simple Prediction Demo")

    st.subheader("1. Overall model performance on full dataset")

    if y_all is not None:
        # Predict on the full dataset (for a simple evaluation)
        y_pred_all = model.predict(X_all)
        mae = mean_absolute_error(y_all, y_pred_all)
        rmse = math.sqrt(mean_squared_error(y_all, y_pred_all))

        col1, col2 = st.columns(2)
        col1.metric("MAE (lower is better)", f"{mae:.4f}")
        col2.metric("RMSE (lower is better)", f"{rmse:.4f}")

        st.write(
            """
            These metrics show how far, on average, the model's predictions are
            from the actual visitor counts.
            """
        )
    else:
        st.warning("Target column not found, cannot compute MAE/RMSE.")

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
        st.write(f"**Predicted visitors:** {y_pred_row:.3f}")
        st.write(f"**Actual visitors:** {y_true_row:.3f}")
    else:
        st.write(f"**Predicted visitors:** {y_pred_row:.3f}")
        st.write("Actual visitors not available (no target column).")


# -----------------------------
# XAI PAGE
# -----------------------------
def show_xai():
    st.title("Explainable AI (XAI) – Banff Visitors Model")

    if y_all is None:
        st.warning("Target column not found, cannot compute XAI plots.")
        return

    # For XAI, drop leakage / shortcut features like today's visits
    X_xai = X_all.drop(columns=[c for c in LEAKAGE_COLS if c in X_all.columns])

    st.markdown(
        """
        The goal of this page is to **explain how the model makes decisions**.

        - We use **global XAI** methods to understand behaviour across the whole dataset.
        - We focus on:
          - Residuals (errors)
          - Global feature importance
          - SHAP global summary
        """
    )

    # 1) Residual plot
    st.subheader("1. Residual Plot (Global Error Behaviour)")
    fig_resid = plot_residuals(model, X_xai, y_all)
    st.pyplot(fig_resid)

    st.markdown(
        """
        - Each point is one record (day / client) from the dataset.  
        - The y-axis shows the **residual** = actual daily visitors − predicted daily visitors.  
        - Points scattered around the red zero line mean the model is not always over- or under-predicting.  
        - Most residuals are small on the scaled target, which matches the low error (MAE/RMSE) we saw earlier.
        """
    )

    st.markdown("---")

    # 2) Permutation feature importance (global)
    st.subheader("2. Global Feature Importance (Permutation Importance)")
    fig_perm = plot_feature_importance(model, X_xai, y_all, top_n=15)
    st.pyplot(fig_perm)

    st.markdown(
        """
        - This is a **global XAI method**: it explains which features the model
          relies on most *on average* across all records.  
        - For each feature, we randomly shuffle its values and measure how much the
          model’s performance gets worse.  
        - Features with longer bars cause a larger drop in performance when shuffled,
          so they are **more important** to the model.  
        - In our Banff project, this lets us:
          - Check that the model uses sensible signals (weekend, month, lag features, etc.).  
          - Detect shortcuts or leakage (for example, if `daily_visits` dominates importance, the model might just copy today’s value to predict tomorrow).
        """
    )

    st.markdown("---")

    # 3) SHAP summary (global)
    st.subheader("3. Global SHAP Summary (Top Features)")

    st.markdown(
        """
        - SHAP is another **global XAI technique** based on Shapley values from game theory.  
        - It shows how much each feature typically pushes predictions **up or down**.  
        - We compute SHAP values on a sample of the dataset for speed.
        """
    )

    try:
        with st.spinner("Computing SHAP explanations on a sample of the data..."):
            fig_shap = plot_shap_summary(model, X_xai)
        st.pyplot(fig_shap)

        st.markdown(
            """
            - The bar plot shows the average absolute SHAP value for each feature.  
            - Features with larger bars have a stronger overall impact on the predicted daily visitors.  
            - For the Banff model, this helps us confirm that important drivers such as:
              - **Weekend vs weekday**
              - **Month / season**
              - **Recent visit history (lag features)**
              are influencing the predictions in a sensible way.
            """
        )
    except Exception as e:
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
        This chatbot uses a simple Retrieval-Augmented Generation (RAG)-style pipeline.

        It:
        - Converts the Banff dataset into short text descriptions.
        - Uses TF-IDF to retrieve the most relevant example days for your question.
        - Builds a small numeric summary (months / weekends / holidays) when your question
          is about those topics.
        - Combines the summary and examples into a natural-language answer.

        Everything runs locally in this app – no external AI API is called.
        """
    )

    # --- Initialize session state BEFORE widgets ---
    if "rag_question" not in st.session_state:
        st.session_state.rag_question = ""
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    # Text input linked to session_state key
    st.text_input(
        "Ask a question about the Banff visitor data:",
        placeholder="e.g., Do weekends have more visitors than weekdays?",
        key="rag_question",
    )

    if st.button("Get answer"):
        q = (st.session_state.rag_question or "").strip()
        if q:
            with st.spinner("Thinking..."):
                answer = rag_answer(q, df)

            # Save to history
            st.session_state.rag_history.append({"q": q, "a": answer})
            # (We do not reset st.session_state.rag_question here.)

    st.subheader("Conversation")

    if not st.session_state.rag_history:
        st.info("Ask a question above to start the conversation.")
    else:
        # Show all previous Q&A (most recent at bottom)
        for item in st.session_state.rag_history:
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(f"**A:** {item['a']}")
            st.markdown("---")


# -----------------------------
# ROUTER
# -----------------------------
if page == "Home":
    show_home()
elif page == "EDA":
    show_eda()
elif page == "Model & Prediction":
    show_model_and_prediction()
elif page == "XAI":
    show_xai()
elif page == "RAG Chatbot":
    show_rag_chatbot()
