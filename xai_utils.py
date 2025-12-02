import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import shap


# ----------------------------------------------------
# Helper: get preprocessor + final estimator from a Pipeline
# ----------------------------------------------------
def _split_pipeline(model):
    """
    If `model` is a sklearn Pipeline, try to separate:
    - preprocessor: anything with transform()
    - final_est: the last step (regressor)
    Otherwise, treat `model` itself as the estimator.
    """
    preprocessor = None
    final_est = model

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.items())
        # assume last step is the final estimator
        final_est = steps[-1][1]
        # look for something that has transform() to use as preprocessor
        for name, step in steps:
            if hasattr(step, "transform"):
                preprocessor = step

    return preprocessor, final_est


# ----------------------------------------------------
# 1. Residual plot
# ----------------------------------------------------
def plot_residuals(model, X, y):
    """
    Make a simple residual plot: residual = actual - predicted.
    Returns a Matplotlib figure.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(residuals)), residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residual Plot")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Residual (actual - predicted)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------
# 2. Permutation feature importance
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
# 3. SHAP summary plot
# ----------------------------------------------------
def plot_shap_summary(model, X):
    """
    SHAP summary plot for the final tree-based model inside a Pipeline.

    - If the model is a Pipeline with a tree-based final estimator,
      use TreeExplainer on the final estimator and the preprocessed features.
    - If that fails, fall back to a simple text-only explanation by
      raising a clear error to the caller.
    """
    preprocessor, final_est = _split_pipeline(model)

    # preprocess X if we have a preprocessor
    if preprocessor is not None:
        X_proc = preprocessor.transform(X)
        # we lose original names here; use generic names
        feature_names = [f"Feature {i}" for i in range(X_proc.shape[1])]
    else:
        X_proc = X.values
        feature_names = list(X.columns)

    # take a sample for speed
    n_samples = min(300, X_proc.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_proc.shape[0], size=n_samples, replace=False)
    X_sample = X_proc[sample_idx]

    try:
        # TreeExplainer works for tree-based models (XGBoost, RandomForest, etc.)
        explainer = shap.TreeExplainer(final_est)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        # Let the caller decide how to show this in the app
        raise RuntimeError(
            f"SHAP TreeExplainer does not support this model type: {type(final_est)}. "
            f"Original error: {e}"
        )

    # Create SHAP summary plot and return the figure
    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="bar",  # bar summary is simpler for presentation
        max_display=15,
    )
    plt.title("Top Features (SHAP Summary)")
    plt.tight_layout()
    return fig
