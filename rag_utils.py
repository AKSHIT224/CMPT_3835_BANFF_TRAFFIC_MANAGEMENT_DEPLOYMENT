from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1. Helpers to turn each row into a short text description
# -------------------------------------------------------------------


def _month_name(val) -> str:
    """Convert a numeric month to a readable name, if possible."""
    month_map = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    try:
        num = int(val)
        return month_map.get(num, str(val))
    except Exception:
        return str(val)


def _bool_flag_to_text(val, weekend: bool = True) -> str:
    """Map 0/1 flags to friendly text for weekend / holiday."""
    true_labels = {1, 1.0, "1", "True", "true", True}

    if weekend:
        return "weekend" if val in true_labels else "weekday"
    else:
        return "holiday" if val in true_labels else "non-holiday"


def _safe_number(val):
    """Return val rounded if numeric, otherwise 'unknown'."""
    try:
        if pd.isna(val):
            return "unknown"
        return round(float(val), 1)
    except Exception:
        return "unknown"


def _row_to_text(row: pd.Series, idx: int) -> str:
    """
    Turn a single row into a short natural-language description.
    We handle missing columns safely with .get().
    """
    month_raw = row.get("month", "unknown")
    month_txt = _month_name(month_raw)

    dow = row.get("day_of_week", "unknown weekday")

    is_weekend_raw = row.get("is_weekend", "unknown")
    is_holiday_raw = row.get("is_holiday", "unknown")

    weekend_txt = (
        _bool_flag_to_text(is_weekend_raw, weekend=True)
        if is_weekend_raw != "unknown"
        else "unknown weekend/weekday"
    )
    holiday_txt = (
        _bool_flag_to_text(is_holiday_raw, weekend=False)
        if is_holiday_raw != "unknown"
        else "unknown holiday flag"
    )

    visitors = _safe_number(row.get("daily_visits.1", row.get("daily_visits", "unknown")))
    rolling_7 = _safe_number(row.get("rolling_7", "unknown"))
    lag_7 = _safe_number(row.get("lag_7", "unknown"))
    lag_30 = _safe_number(row.get("lag_30", "unknown"))

    return (
        f"Row {idx}: A day in {month_txt} ({dow}; {weekend_txt}, {holiday_txt}), "
        f"visitors = {visitors}, rolling_7 = {rolling_7}, "
        f"lag_7 = {lag_7}, lag_30 = {lag_30}."
    )


def build_documents(df: pd.DataFrame, max_rows: int = 500) -> list[str]:
    """
    Build a list of text documents (one per row) from the dataframe.
    We limit to `max_rows` to keep things light.
    """
    if df is None or df.empty:
        return []

    sample = df.head(max_rows)
    documents: list[str] = []

    for i, (_, row) in enumerate(sample.iterrows()):
        documents.append(_row_to_text(row, i))

    return documents


# -------------------------------------------------------------------
# 2. TF-IDF index + cosine similarity for retrieval
# -------------------------------------------------------------------

# Simple in-memory cache so we don't rebuild vectorizer on every call
_DOCS_CACHE: list[str] | None = None
_VECTORIZER_CACHE: TfidfVectorizer | None = None
_DOC_MATRIX_CACHE = None


def _ensure_index(df: pd.DataFrame) -> None:
    """
    Build TF-IDF index for the dataset if not already built.
    Uses module-level globals as a tiny cache.
    """
    global _DOCS_CACHE, _VECTORIZER_CACHE, _DOC_MATRIX_CACHE

    if (
        _DOCS_CACHE is not None
        and _VECTORIZER_CACHE is not None
        and _DOC_MATRIX_CACHE is not None
    ):
        return  # already built

    docs = build_documents(df)
    if not docs:
        _DOCS_CACHE = []
        _VECTORIZER_CACHE = None
        _DOC_MATRIX_CACHE = None
        return

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(docs)

    _DOCS_CACHE = docs
    _VECTORIZER_CACHE = vectorizer
    _DOC_MATRIX_CACHE = doc_matrix


def retrieve_context(query: str, df: pd.DataFrame, top_k: int = 5) -> list[str]:
    """
    Given a free-text query, return the top_k most similar document strings.
    """
    _ensure_index(df)

    if not _DOCS_CACHE or _VECTORIZER_CACHE is None or _DOC_MATRIX_CACHE is None:
        return []

    q_vec = _VECTORIZER_CACHE.transform([query])
    sims = cosine_similarity(q_vec, _DOC_MATRIX_CACHE)[0]

    # Highest similarity first
    top_idx = np.argsort(sims)[::-1][:top_k]
    results: list[str] = []

    for i in top_idx:
        # Skip if similarity is zero (no overlap)
        if sims[i] <= 0:
            continue
        results.append(_DOCS_CACHE[i])

    return results


# -------------------------------------------------------------------
# 3. Simple aggregate summaries (weekends / rolling_7 / lag features)
# -------------------------------------------------------------------


def _safe_group_mean(df: pd.DataFrame, value_col: str, group_col: str):
    """Return mean visitors by group_col, or None if columns missing."""
    if value_col not in df.columns or group_col not in df.columns:
        return None
    gb = (
        df[[group_col, value_col]]
        .dropna()
        .groupby(group_col)[value_col]
        .mean()
        .sort_values(ascending=False)
    )
    if gb.empty:
        return None
    return gb


def _subset_for_april_if_mentioned(q: str, df: pd.DataFrame) -> pd.DataFrame:
    """If user says 'April', focus on month==4, otherwise use full df."""
    if "april" in q and "month" in df.columns:
        sub = df[df["month"] == 4]
        if len(sub) > 0:
            return sub
    return df


def _build_aggregate_summary(query: str, df: pd.DataFrame) -> str:
    """
    Build a short numeric summary based on the type of question.
    Handles:
      - weekend vs weekday
      - rolling 7-day averages
      - standout busy days vs rolling_7
      - lag features (lag_7, lag_14, lag_30)
    """
    if df is None or df.empty:
        return ""

    q = query.lower()

    # Choose value column
    value_col = "daily_visits.1" if "daily_visits.1" in df.columns else "daily_visits"
    if value_col not in df.columns:
        return ""

    base_df = _subset_for_april_if_mentioned(q, df)
    parts: list[str] = []

    # A) Weekends vs weekdays (optionally only April)
    if any(k in q for k in ["weekend", "weekday"]):
        by_weekend = _safe_group_mean(base_df, value_col, "is_weekend")
        if by_weekend is not None:
            d = by_weekend.to_dict()
            weekend_val = d.get(1, d.get(1.0, None))
            weekday_val = d.get(0, d.get(0.0, None))
            if weekend_val is not None and weekday_val is not None:
                parts.append(
                    f"On average, weekends have about {weekend_val:.0f} visitors/day "
                    f"versus {weekday_val:.0f} visitors/day on weekdays "
                    f"in this subset of the data."
                )

    # B) Rolling-7 behaviour (trend vs actual)
    if any(k in q for k in ["rolling", "7-day", "7 day"]):
        if "rolling_7" in base_df.columns:
            valid = base_df[[value_col, "rolling_7"]].dropna()
            if len(valid) > 0:
                mean_vis = valid[value_col].mean()
                mean_roll = valid["rolling_7"].mean()
                diff = valid[value_col] - valid["rolling_7"]
                mean_diff = diff.mean()
                if mean_diff > 0:
                    relation = "above"
                elif mean_diff < 0:
                    relation = "below"
                else:
                    relation = "in line with"
                parts.append(
                    "The 7-day rolling average smooths daily ups and downs. "
                    f"In this subset, actual visitors average about {mean_vis:.0f} per day, "
                    f"while the rolling average is around {mean_roll:.0f}. "
                    f"On a typical day, actual counts are about {abs(mean_diff):.0f} visitors "
                    f"{relation} the rolling trend."
                )

    # C) Standout busy days compared to rolling_7
    if any(k in q for k in ["stand out", "unusually", "much busier"]):
        if "rolling_7" in base_df.columns:
            valid = base_df[[value_col, "rolling_7"]].dropna()
            if len(valid) > 0:
                diff = valid[value_col] - valid["rolling_7"]
                top = diff.nlargest(3)
                parts.append(
                    "Some days have visitor counts far above the recent 7-day trend. "
                    "For example, the three biggest jumps are about "
                    + ", ".join(f"{v:.0f} visitors above the rolling average" for v in top.values)
                    + "."
                )

    # D) Lag behaviour (lag_7, lag_14, lag_30)
    if "lag" in q:
        lag_sentences: list[str] = []
        for lag_col, label in [
            ("lag_7", "7 days ago"),
            ("lag_14", "14 days ago"),
            ("lag_30", "30 days ago"),
        ]:
            if lag_col in base_df.columns:
                valid = base_df[[value_col, lag_col]].dropna()
                if len(valid) > 5:
                    corr = valid[value_col].corr(valid[lag_col])
                    lag_sentences.append(
                        f"Visitor counts have a correlation of about {corr:.2f} "
                        f"with values from {label}."
                    )
        if lag_sentences:
            parts.append("Looking at lag features, we see that " + " ".join(lag_sentences))

    if not parts:
        return ""

    return "### Short data-based summary\n\n" + " ".join(parts) + "\n\n"


# -------------------------------------------------------------------
# 4. "Answer" builder (summaries + example rows, no external LLM)
# -------------------------------------------------------------------


def _build_answer_from_context(
    query: str, context_docs: list[str], df: pd.DataFrame
) -> str:
    """
    Build a simple human-readable answer string from:
      - small numeric summary (optional)
      - retrieved example rows
    """
    summary = _build_aggregate_summary(query, df)

    if not context_docs and not summary:
        return (
            "I could not find any rows in the dataset that match your question well. "
            "Try asking in simpler words (for example: "
            "'Do weekends show higher visitor numbers than weekdays in April?')."
        )

    bullets = "\n".join(f"- {doc}" for doc in context_docs)
    answer_parts: list[str] = []

    if summary:
        answer_parts.append(summary)

    if context_docs:
        answer_parts.append("### Example days from the dataset\n\n")
        answer_parts.append("You asked:\n\n")
        answer_parts.append(f"**{query}**\n\n")
        answer_parts.append("Here are some relevant example days from the Banff dataset:\n\n")
        answer_parts.append(f"{bullets}\n\n")
        answer_parts.append(
            "From these examples, you can see how factors like month, "
            "weekend/holiday status, and recent visitor trends relate to the "
            "number of visitors on those days."
        )

    return "".join(answer_parts)


# -------------------------------------------------------------------
# 5. Public function used by app.py
# -------------------------------------------------------------------


def rag_answer(query: str, df: pd.DataFrame) -> str:
    """
    Main entry point called by app.py.

    - Takes the user's question and the Banff dataframe
    - Computes simple summary stats when relevant
    - Retrieves similar rows using TF-IDF
    - Returns a plain-text answer built from those pieces
    """
    if not isinstance(query, str) or not query.strip():
        return "Please type a non-empty question about the Banff visitor data."

    context_docs = retrieve_context(query, df, top_k=5)
    answer = _build_answer_from_context(query, context_docs, df)
    return answer
