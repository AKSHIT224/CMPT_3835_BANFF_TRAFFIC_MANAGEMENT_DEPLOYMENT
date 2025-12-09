import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap
from scipy import sparse


# ----------------------------------------------------
# Helper: extract preprocessor and model from Pipeline
# ----------------------------------------------------
def _split_pipeline(model):
    """
    If model is a Pipeline:
      - return (preprocessor, final_estimator)
    Otherwise:
      - return (None, model)
    """
    preprocessor = None
    final_est = model

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        final_est = steps[-1][1]  # last step = estimator
        # assume the last transform step before estimator is the preprocessor
        for name, step in steps[:-1]:
            if hasattr(step, "transform"):
                preprocessor = step

    return preprocessor, final_est


# ----------------------------------------------------
# 1. Residual Plot
# ----------------------------------------------------
def plot_residuals(model, X, y, clip_quantile=0.99):
    """
    Global error behaviour: residual = actual - predicted.
    Residuals are plotted against predicted values.
    To avoid distortion by extreme outliers, we show only
    the central `clip_quantile` fraction of residuals.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    # clip extreme residuals for nicer visualisation
    abs_resid = np.abs(residuals)
    limit = np.quantile(abs_resid, clip_quantile)
    mask = abs_resid <= limit

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred[mask], residuals[mask], alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual Plot (Actual - Predicted, central 99% of residuals)")
    ax.set_xlabel("Predicted visitors (scaled)")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 2. Permutation Feature Importance
# ----------------------------------------------------
def plot_feature_importance(model, X, y, top_n=8, min_importance=None):
    """
    Global feature importance using permutation importance
    on a subset of the data (faster).

    Shows up to `top_n` most important features. Optionally
    drops features whose importance is below `min_importance`.
    """

    # sample for faster XAI computation
    if len(X) > 2000:
        sample = X.sample(n=2000, random_state=42)
        y_sample = y.loc[sample.index]
    else:
        sample = X
        y_sample = y

    try:
        result = permutation_importance(
            model,
            sample,
            y_sample,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        return None

    importances = result.importances_mean

    # sort by importance (descending)
    idx_sorted = np.argsort(importances)[::-1]

    if min_importance is not None:
        idx_sorted = [i for i in idx_sorted if importances[i] >= min_importance]

    # ensure we have something to plot
    if len(idx_sorted) == 0:
        idx_sorted = np.argsort(importances)[::-1]

    idx_final = idx_sorted[:top_n]
    sorted_importances = importances[idx_final]
    feature_names = np.array(sample.columns)[idx_final]

    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(sorted_importances)  # small to large, for bottom-to-top bars
    ax.barh(range(len(sorted_importances)), sorted_importances[order])
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(feature_names[order])
    ax.set_xlabel("Permutation importance (decrease in model performance)")
    ax.set_title("Main Global Features (Permutation Importance)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 3. SHAP Summary Plot
# ----------------------------------------------------
def plot_shap_summary(model, X, max_display=8):
    """
    SHAP summary plot for models inside a Pipeline.

    Uses shap.Explainer on the final estimator and the preprocessed X.
    Returns a matplotlib Figure or None if SHAP cannot be computed.
    """

    preprocessor, final_est = _split_pipeline(model)

    # preprocess X if needed
    if preprocessor is not None:
        X_proc = preprocessor.transform(X)
        # try to get proper feature names from the preprocessor
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            # fall back to original column names if available
            feature_names = list(getattr(X, "columns", [f"Feature {i+1}" for i in range(X_proc.shape[1])]))
    else:
        X_proc = X
        feature_names = list(getattr(X, "columns", [f"Feature {i+1}" for i in range(X.shape[1])]))

    # convert to dense numpy array if needed
    if sparse.issparse(X_proc):
        X_proc = X_proc.toarray()
    else:
        X_proc = np.asarray(X_proc)

    # small sample for speed
    n_samples = min(400, X_proc.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(X_proc.shape[0], n_samples, replace=False)
    X_sample = X_proc[idx]

    try:
        explainer = shap.Explainer(final_est, X_sample)
        shap_values = explainer(X_sample)
    except Exception:
        # let the Streamlit app show a friendly message if this returns None
        return None

    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=max_display,
    )
    plt.title("Top Global Features (SHAP Summary)")
    plt.tight_layout()
    return fig
