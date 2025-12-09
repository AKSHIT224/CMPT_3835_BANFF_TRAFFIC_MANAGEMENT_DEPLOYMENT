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
