import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap


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
        final_est = steps[-1][1]
        for name, step in steps:
            if hasattr(step, "transform"):
                preprocessor = step

    return preprocessor, final_est


# ----------------------------------------------------
# 1. Residual Plot
# ----------------------------------------------------
def plot_residuals(model, X, y):
    """
    Global error behaviour: residual = actual - predicted.
    Residuals are plotted against predicted values.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual Plot (Actual - Predicted)")
    ax.set_xlabel("Predicted visitors (scaled)")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 2. Permutation Feature Importance
# ----------------------------------------------------
def plot_feature_importance(model, X, y, top_n=15):
    """
    Global feature importance using permutation importance
    on a subset of the data (faster).
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

    # sort & select top features
    idx = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[idx]
    feature_names = np.array(sample.columns)[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_importances)), sorted_importances[::-1])
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(feature_names[::-1])
    ax.set_xlabel("Permutation importance (decrease in model performance)")
    ax.set_title("Top Global Features (Permutation Importance)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 3. SHAP Summary Plot
# ----------------------------------------------------
def plot_shap_summary(model, X):
    """
    SHAP summary plot for tree-based models inside a Pipeline.
    Uses TreeExplainer on the final estimator and the preprocessed X.
    """

    preprocessor, final_est = _split_pipeline(model)

    # preprocess X if needed
    if preprocessor is not None:
        X_proc = preprocessor.transform(X)
        # Generate generic names because OneHotEncoder expands columns
        feature_names = [f"Feature {i}" for i in range(X_proc.shape[1])]
    else:
        X_proc = X.values
        feature_names = list(X.columns)

    # take small sample for speed
    n_samples = min(300, X_proc.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(X_proc.shape[0], n_samples, replace=False)
    X_sample = X_proc[idx]

    try:
        explainer = shap.TreeExplainer(final_est)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        raise RuntimeError("SHAP could not be computed for this model.")

    # SHAP bar summary
    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=15,
    )
    plt.title("Top Global Features (SHAP Summary)")
    plt.tight_layout()
    return fig
