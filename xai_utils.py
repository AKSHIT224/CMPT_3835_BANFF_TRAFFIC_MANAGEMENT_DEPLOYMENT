import matplotlib.pyplot as plt
import numpy as np
import shap

# -------------------------
# 1. RESIDUAL PLOT
# -------------------------
def plot_residuals(model, X, y):
    y_pred = model.predict(X)
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(residuals)), residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Residual (actual - predicted)")
    return fig


# -------------------------
# 2. CLEAN FEATURE IMPORTANCE (XGBOOST BUILT-IN)
# -------------------------
def plot_feature_importance(model, X, y):
    """
    Uses model.get_booster().get_score() to show clean feature importance.
    Works perfectly with XGBoost models.
    """
    try:
        importance_dict = model.get_booster().get_score(importance_type="gain")
    except:
        return None

    # Convert to a sorted list
    labels, scores = zip(*sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, scores)
    ax.set_title("Feature Importance (XGBoost - Gain)")
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    return fig


# -------------------------
# 3. CLEAN SHAP SUMMARY PLOT
# -------------------------
def plot_shap_summary(model, X):
    """
    Correct SHAP for XGBoost:
    - Uses TreeExplainer
    - Shows real feature names
    """
    # Use a small sample to speed up SHAP
    sample = X.sample(n=min(300, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, sample, show=False)
    return fig
