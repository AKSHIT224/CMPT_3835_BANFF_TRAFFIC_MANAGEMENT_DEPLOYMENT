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

        Data source: final_selected_features.csv  
        Best model: tuned XGBoost (saved as best_model.pkl)
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
            These metrics show how far, on average, the model's predictions are from the actual visitor counts.
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
        step=1
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
    st.title("Explainable AI (XAI)")

    if y_all is None:
        st.warning("Target column not found, cannot compute XAI plots.")
        return

    st.write(
        """
        These plots help explain **how** the model makes its predictions.

        - **Residual plot:** shows where the model over- or under-predicts.
        - **Feature importance:** which features matter most overall.
        - **SHAP summary:** how each feature pushes predictions up or down for many samples.
        """
    )

    # 1) Residual plot
    st.subheader("1. Residual plot (actual - predicted)")
    fig_res = plot_residuals(model, X_all, y_all)
    st.pyplot(fig_res)

    st.write(
        """
        Points close to zero mean the prediction is very accurate.  
        Large positive or negative residuals show days where the model struggled more.
        """
    )

    # 2) Feature importance
    st.subheader("2. Global feature importance")
    fig_imp = plot_feature_importance(model, X_all, y_all)
    st.pyplot(fig_imp)

    st.write(
        """
        Features at the top of this chart have the strongest overall impact on the prediction.  
        This helps you see which variables are most important for estimating daily visitors.
        """
    )

    # 3) SHAP summary
    st.subheader("3. SHAP summary plot (sample of rows)")
    st.write("This plot shows how each feature affects the prediction across many samples.")

    fig_shap = plot_shap_summary(model, X_all)
    st.pyplot(fig_shap)


# -----------------------------
# RAG CHATBOT PAGE
# -----------------------------
def show_rag_chatbot():
    st.title("RAG-based Chatbot – Banff Visitors")

    st.write(
        """
        This chatbot uses a simple Retrieval-Augmented Generation (RAG) pipeline.

        It:
        - Converts the Banff dataset into short text descriptions.
        - Retrieves the most relevant pieces of text for your question.
        - Uses a language model (Gemini 1.5) to answer based on that context.

        You can ask general questions about patterns in the Banff visitor data.
        """
    )

    # Initialize chat history in session state
    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    # Text input with a key so we can clear it after each question
    user_question = st.text_input(
        "Ask a question about the Banff visitor data:",
        placeholder="e.g., Which features are important for high visitor days?",
        key="rag_question",
    )

    if st.button("Get answer"):
        q = st.session_state.rag_question.strip()
        if q:
            with st.spinner("Thinking..."):
                answer = rag_answer(q, df)

            # Save to history
            st.session_state.rag_history.append({"q": q, "a": answer})
            # Clear the input box for the next question
            st.session_state.rag_question = ""

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
