import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import shap
from sklearn.inspection import permutation_importance


def _sample_X(X: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
    """
    Return a random sample of rows from X (for speed in XAI plots).
    """
    if len(X) > max_samples:
        return X.sample(max_samples, random_state=42)
    return X


def plot_residuals(model, X: pd.DataFrame, y: pd.Series, max_points: int = 2000):
    """
    Make a residual plot (actual - predicted) for a sample of the data.
    Returns a matplotlib figure.
    """
    X_sample = _sample_X(X, max_points)
    y_sample = y.loc[X_sample.index]

    y_pred = model.predict(X_sample)
    residuals = y_sample - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(len(residuals)), residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual Plot")

    return fig


def plot_feature_importance(model, X: pd.DataFrame, y: pd.Series, top_n: int = 15):
    """
    Plot global feature importance using permutation importance.

    This works directly with Pipelines (preprocessing + model)
    and keeps the original feature names.
    """
    X_sample = _sample_X(X, 1000)
    y_sample = y.loc[X_sample.index]

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_squared_error",
    )

    importances_mean = result.importances_mean
    imp_series = pd.Series(importances_mean, index=X_sample.columns)
    imp_series = imp_series.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_series.index[::-1], imp_series.values[::-1])
    ax.set_xlabel("Importance (permutation decrease in performance)")
    ax.set_title(f"Top {top_n} Feature Importances (Permutation)")
    plt.tight_layout()
    return fig


def plot_shap_summary(model, X: pd.DataFrame, max_samples: int = 500):
    """
    Plot a SHAP summary (beeswarm) for a sample of rows.

    For a Pipeline(model=tree_model, prep=ColumnTransformer),
    we:
      - transform X with the preprocessor
      - run TreeExplainer on the inner tree model
    """
    X_sample = _sample_X(X, max_samples)

    # Handle Pipeline with 'prep' and 'model' steps
    inner_model = model
    X_for_shap = X_sample

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        inner_model = model.named_steps["model"]
        if "prep" in model.named_steps:
            prep = model.named_steps["prep"]
            X_for_shap = prep.transform(X_sample)
        else:
            X_for_shap = X_sample

    # Use TreeExplainer for tree-based models (like XGBoost)
    explainer = shap.TreeExplainer(inner_model)
    shap_values = explainer(X_for_shap)

    # SHAP creates its own figure; we grab the current one
    shap.summary_plot(shap_values, X_for_shap, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)

    return fig
