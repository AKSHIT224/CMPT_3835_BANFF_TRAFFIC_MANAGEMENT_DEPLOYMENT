from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------
# 0. Small helpers
# -------------------------------------------------------------------


def _get_target_col(df: pd.DataFrame) -> str | None:
    """Try to find the visitor/target column."""
    for c in ["daily_visits.1", "daily_visits", "visits"]:
        if c in df.columns:
            return c
    return None


def _safe_mean(series: pd.Series) -> float | None:
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


# -------------------------------------------------------------------
# 1. Convert each row into a short text description  (for examples)
# -------------------------------------------------------------------


def _row_to_text(row: pd.Series, idx: int) -> str:
    """
    Turn a single row into a short natural-language description.
    We handle missing columns safely with .get().
    """
    date = row.get("date", "unknown date")
    month = row.get("month", "unknown month")
    dow = row.get("day_of_week", "unknown weekday")
    is_weekend = row.get("is_weekend", "unknown")
    is_holiday = row.get("is_holiday", "unknown")

    visitors = row.get("daily_visits.1", row.get("daily_visits", "unknown"))
    rolling_7 = row.get("rolling_7", "unknown")
    lag_1 = row.get("lag_1", "unknown")
    lag_7 = row.get("lag_7", "unknown")

    return (
        f"Row {idx}: On {date} (a day in {month}, {dow}; "
        f"weekend={is_weekend}, holiday={is_holiday}), "
        f"the visitors were {visitors}. "
        f"The rolling 7-day average was {rolling_7}, "
        f"lag_1 was {lag_1}, and lag_7 was {lag_7}."
    )


def build_documents(df: pd.DataFrame, max_rows: int = 500) -> list[str]:
    """
    Build a list of text documents (one per row) from the dataframe.
    We limit to `max_rows` to keep things light.
    """
    if df is None or df.empty:
        return []

    sample = df.head(max_rows)
    documents = []

    for i, (_, row) in enumerate(sample.iterrows()):
        documents.append(_row_to_text(row, i))

    return documents


# -------------------------------------------------------------------
# 2. TF-IDF index + cosine similarity for retrieval  (for examples)
# -------------------------------------------------------------------

_DOCS_CACHE: list[str] | None = None
_VECTORIZER_CACHE: TfidfVectorizer | None = None
_DOC_MATRIX_CACHE = None


def _ensure_index(df: pd.DataFrame):
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
    results = []

    for i in top_idx:
        # Skip if similarity is zero (no overlap)
        if sims[i] <= 0:
            continue
        results.append(_DOCS_CACHE[i])

    return results


# -------------------------------------------------------------------
# 3. Small analytic summaries for common question types  (Option A)
# -------------------------------------------------------------------


def _detect_question_type(query: str) -> str:
    """
    Very simple rules to decide what kind of question this is.
    """
    q = query.lower()

    if "month" in q and ("busy" in q or "busiest" in q or "season" in q or "peak" in q):
        return "busy_months"

    if ("weekend" in q or "week day" in q or "weekday" in q) and (
        "more" in q
        or "less" in q
        or "higher" in q
        or "lower" in q
        or "difference" in q
    ):
        return "weekend_vs_weekday"

    if "holiday" in q and (
        "more" in q or "less" in q or "effect" in q or "impact" in q or "influence" in q
    ):
        return "holiday_effect"

    return "generic"


def _summary_busy_months(df: pd.DataFrame) -> str | None:
    target_col = _get_target_col(df)
    if target_col is None:
        return None

    # Try text month name first, then numeric month
    month_col = None
    for c in ["month_name", "month", "month_num"]:
        if c in df.columns:
            month_col = c
            break

    if month_col is None:
        return None

    g = df[[month_col, target_col]].copy()
    g[target_col] = pd.to_numeric(g[target_col], errors="coerce")
    g = g.dropna(subset=[target_col])

    if g.empty:
        return None

    # If month is numeric, map to name just for display
    if np.issubdtype(g[month_col].dtype, np.number):
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
        g["month_display"] = g[month_col].map(month_map).fillna(g[month_col].astype(str))
    else:
        g["month_display"] = g[month_col].astype(str)

    avg_by_month = (
        g.groupby("month_display")[target_col].mean().sort_values(ascending=False)
    )

    if avg_by_month.empty:
        return None

    top = avg_by_month.head(3)
    parts = []
    for month_name, val in top.items():
        parts.append(f"{month_name} (~{int(round(val))} visitors/day)")

    months_list = ", ".join(parts)
    return (
        "### Short data-based summary\n\n"
        "Based on the dataset, the months with the highest average daily visitors are:\n\n"
        f"{months_list}. These months can be considered the 'busy season' in this dataset."
    )


def _summary_weekend_vs_weekday(df: pd.DataFrame) -> str | None:
    target_col = _get_target_col(df)
    if target_col is None or "is_weekend" not in df.columns:
        return None

    weekend = _safe_mean(df.loc[df["is_weekend"] == 1, target_col])
    weekday = _safe_mean(df.loc[df["is_weekend"] == 0, target_col])

    if weekend is None or weekday is None:
        return None

    weekend = int(round(weekend))
    weekday = int(round(weekday))

    if weekend > weekday:
        trend = "On average, **weekends are busier than weekdays**."
    elif weekend < weekday:
        trend = "On average, **weekdays are busier than weekends**."
    else:
        trend = "On average, **weekends and weekdays have very similar visitor levels**."

    return (
        "### Short data-based summary\n\n"
        f"Average visitors on weekends: ~{weekend} per day.\n\n"
        f"Average visitors on weekdays: ~{weekday} per day.\n\n"
        f"{trend}"
    )


def _summary_holiday_effect(df: pd.DataFrame) -> str | None:
    target_col = _get_target_col(df)
    if target_col is None or "is_holiday" not in df.columns:
        return None

    holiday = _safe_mean(df.loc[df["is_holiday"] == 1, target_col])
    non_holiday = _safe_mean(df.loc[df["is_holiday"] == 0, target_col])

    if holiday is None or non_holiday is None:
        return None

    holiday = int(round(holiday))
    non_holiday = int(round(non_holiday))

    if holiday > non_holiday:
        trend = "On average, **holidays are busier than normal days**."
    elif holiday < non_holiday:
        trend = "On average, **normal days are busier than holidays**."
    else:
        trend = "On average, **holidays and non-holidays have very similar visitor levels**."

    return (
        "### Short data-based summary\n\n"
        f"Average visitors on holidays: ~{holiday} per day.\n\n"
        f"Average visitors on non-holidays: ~{non_holiday} per day.\n\n"
        f"{trend}"
    )


def _make_numeric_summary(query: str, df: pd.DataFrame) -> str | None:
    qtype = _detect_question_type(query)

    if qtype == "busy_months":
        return _summary_busy_months(df)
    if qtype == "weekend_vs_weekday":
        return _summary_weekend_vs_weekday(df)
    if qtype == "holiday_effect":
        return _summary_holiday_effect(df)

    return None  # generic question → no numeric summary


# -------------------------------------------------------------------
# 4. Build the answer string (summary + examples)  (Option A + B)
# -------------------------------------------------------------------


def _format_examples_section(query: str, context_docs: list[str]) -> str:
    """Build the 'example rows' section of the answer."""
    if not context_docs:
        return (
            "I could not find any rows in the dataset that match your question well. "
            "Try asking in simpler words (for example: 'Which months are busy?' "
            "or 'Do weekends have more visitors than weekdays?')."
        )

    bullets = "\n".join(f"- {doc}" for doc in context_docs)

    return (
        "### Example days from the dataset\n\n"
        "You asked:\n\n"
        f"**{query}**\n\n"
        "Here are some relevant example days from the Banff dataset:\n\n"
        f"{bullets}\n\n"
        "From these examples, you can see how factors like month, "
        "weekend/holiday status, and recent visitor trends relate to the "
        "number of visitors on those days."
    )


# -------------------------------------------------------------------
# 5. Public function used by app.py
# -------------------------------------------------------------------


def rag_answer(query: str, df: pd.DataFrame) -> str:
    """
    Main entry point called by app.py.

    - Takes the user's question and the Banff dataframe
    - (A) Tries to compute a small numeric summary for common question types
    - (B) Retrieves similar rows using TF-IDF and shows example days
    """
    if not isinstance(query, str) or not query.strip():
        return "Please type a non-empty question about the Banff visitor data."

    # (A) numeric summary (if we can detect a specific pattern)
    summary = _make_numeric_summary(query, df)

    # (B) RAG-style examples
    context_docs = retrieve_context(query, df, top_k=5)
    examples = _format_examples_section(query, context_docs)

    if summary:
        return f"{summary}\n\n{examples}"
    else:
        # generic question → only examples
        return examples