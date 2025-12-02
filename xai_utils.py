import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap


# ----------------------------------------------------
# 1. Residual plot (global error behaviour)
# ----------------------------------------------------
def plot_residuals(model, X, y):
    """
    Global error behaviour: residual = actual - predicted.
    Residuals are plotted against predicted values.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, s=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual Plot (Global Error Behaviour)")
    ax.set_xlabel("Predicted daily visits (scaled)")
    ax.set_ylabel("Residual (actual - predicted)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 2. Permutation feature importance (global XAI)
# ----------------------------------------------------
def plot_feature_importance(model, X, y, top_n: int = 15):
    """
    Global feature importance using permutation importance
    on a sample of the data (for speed).

    Returns a Matplotlib figure with a horizontal bar chart.
    """
    # sample to keep it fast
    if len(X) > 2000:
        sample = X.sample(n=2000, random_state=42)
        y_sample = y.loc[sample.index]
    else:
        sample = X
        y_sample = y

    result = permutation_importance(
        model,
        sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importances = result.importances_mean
    # sort by importance
    idx = np.argsort(importances)[::-1][:top_n]

    sorted_importances = importances[idx]
    feature_names = np.array(sample.columns)[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_importances)), sorted_importances[::-1])
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(feature_names[::-1])
    ax.set_xlabel("Permutation importance (avg. decrease in performance)")
    ax.set_title("Top Features (Permutation Importance)")
    ax.invert_yaxis()  # biggest at top
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 3. SHAP summary plot (global XAI)
# ----------------------------------------------------
def plot_shap_summary(model, X):
    """
    Global SHAP summary (bar chart).

    Uses shap.Explainer on the full model (including any preprocessing),
    and keeps the original feature names from X.

    IMPORTANT:
    - X should be a pandas DataFrame with the same columns
      that were used during training.
    """
    # take a sample for speed
    sample = X.sample(n=min(300, len(X)), random_state=42)

    # Build a general, model-agnostic explainer around model.predict
    explainer = shap.Explainer(model.predict, sample)
    shap_values = explainer(sample)

    # Global bar plot of mean(|SHAP|) per feature
    fig = plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title("Top Features (SHAP Global Explanation)")
    plt.tight_layout()
    return fig
