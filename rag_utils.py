from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 0. Small helpers
# -------------------------------------------------------------------


def _get_target_col(df: pd.DataFrame) -> str | None:
    """
    Try to find the main 'visitors' target column in the dataframe.
    We check a few common names.
    """
    candidates = ["daily_visits.1", "daily_visits", "visitors", "visitor_count"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


_MONTH_NAMES = {
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

_DOW_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _to_month_name(val) -> str:
    try:
        num = int(val)
        if num in _MONTH_NAMES:
            return _MONTH_NAMES[num]
    except Exception:
        pass
    return str(val)


def _to_dow_name(val) -> str:
    try:
        num = int(val)
        if num in _DOW_NAMES:
            return _DOW_NAMES[num]
    except Exception:
        pass
    return str(val)


def _flag_to_text(val, true_word="yes", false_word="no") -> str:
    if pd.isna(val):
        return "unknown"
    try:
        num = float(val)
        return true_word if num == 1 else false_word
    except Exception:
        return str(val)


# -------------------------------------------------------------------
# 1. Convert each row into a short text description
# -------------------------------------------------------------------


def _row_to_text(row: pd.Series, idx: int) -> str:
    """
    Turn a single row into a short natural-language description.
    We handle missing columns safely with .get().
    """
    date = row.get("date", "unknown date")

    month_raw = row.get("month", "unknown")
    month_name = _to_month_name(month_raw)

    dow_raw = row.get("day_of_week", "unknown")
    dow_name = _to_dow_name(dow_raw)

    is_weekend = _flag_to_text(row.get("is_weekend", np.nan), "weekend", "weekday")
    is_holiday = _flag_to_text(row.get("is_holiday", np.nan), "holiday", "non-holiday")

    visitors = row.get("daily_visits.1", row.get("daily_visits", "unknown"))
    rolling_7 = row.get("rolling_7", "unknown")
    lag_1 = row.get("lag_1", "unknown")
    lag_7 = row.get("lag_7", "unknown")

    return (
        f"Row {idx}: On {date} (a day in {month_name}, {dow_name}; "
        f"{is_weekend}, {is_holiday}), the visitors were {visitors}. "
        f"The rolling 7-day average was {rolling_7}, lag_1 was {lag_1}, "
        f"and lag_7 was {lag_7}."
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

_DOCS_CACHE: list[str] | None = None
_VECTORIZER_CACHE: TfidfVectorizer | None = None
_DOC_MATRIX_CACHE = None


def _ensure_index(df: pd.DataFrame):
    """
    Build TF-IDF index for the dataset if not already built.
    Uses module-level globals as a tiny cache.
    """
    global _DOCS_CACHE, _VECTORIZER_CACHE, _DOC_MATRIX_CACHE

    if _DOCS_CACHE is not None and _VECTORIZER_CACHE is not None and _DOC_MATRIX_CACHE is not None:
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
# 3. Simple analytics answers for common question types (Option B)
# -------------------------------------------------------------------


def _busy_months_answer(df: pd.DataFrame, target_col: str) -> str | None:
    # prefer a numeric month column if present
    if "month" not in df.columns and "month_name" not in df.columns:
        return None

    if "month" in df.columns:
        grp = df.groupby("month")[target_col].mean().dropna()
        if grp.empty:
            return None

        # map to nice month names
        records = []
        for m, val in grp.items():
            name = _to_month_name(m)
            records.append((name, float(val)))
    else:
        grp = df.groupby("month_name")[target_col].mean().dropna()
        if grp.empty:
            return None
        records = [(str(m), float(v)) for m, v in grp.items()]

    # sort by average visitors
    records.sort(key=lambda x: x[1], reverse=True)
    top = records[:3]

    parts = [f"{name} (~{val:.0f} visitors/day)" for name, val in top]
    months_text = ", ".join(parts)

    return (
        "Based on the dataset, the months with the highest average daily visitors are:\n"
        f"- {months_text}.\n"
        "These months can be considered the 'busy season' in Banff in this dataset."
    )


def _weekend_vs_weekday_answer(df: pd.DataFrame, target_col: str) -> str | None:
    if "is_weekend" not in df.columns:
        return None

    grp = df.groupby("is_weekend")[target_col].mean().dropna()
    if grp.empty or len(grp) == 1:
        return None

    weekend_avg = float(grp.get(1, np.nan))
    weekday_avg = float(grp.get(0, np.nan))

    if np.isnan(weekend_avg) or np.isnan(weekday_avg):
        return None

    diff = weekend_avg - weekday_avg
    relation = "higher" if diff > 0 else "lower"
    return (
        f"On average, weekends have about {weekend_avg:.0f} visitors per day, "
        f"while weekdays have about {weekday_avg:.0f}. "
        f"That means weekend days tend to be {abs(diff):.0f} visitors {relation} than weekdays."
    )


def _holiday_answer(df: pd.DataFrame, target_col: str) -> str | None:
    if "is_holiday" not in df.columns:
        return None

    grp = df.groupby("is_holiday")[target_col].mean().dropna()
    if grp.empty or len(grp) == 1:
        return None

    holiday_avg = float(grp.get(1, np.nan))
    nonholiday_avg = float(grp.get(0, np.nan))

    if np.isnan(holiday_avg) or np.isnan(nonholiday_avg):
        return None

    diff = holiday_avg - nonholiday_avg
    relation = "higher" if diff > 0 else "lower"
    return (
        f"In this dataset, holidays have about {holiday_avg:.0f} visitors per day on average, "
        f"while non-holidays have about {nonholiday_avg:.0f}. "
        f"So holidays tend to be {abs(diff):.0f} visitors {relation} than normal days."
    )


def _try_simple_analytics_answer(query: str, df: pd.DataFrame) -> str | None:
    """
    Look at the question text and, for a few common patterns, compute a direct
    data-based answer from the dataframe.
    """
    target_col = _get_target_col(df)
    if target_col is None:
        return None

    q = query.lower()

    # Which months are busy / busiest?
    if "month" in q and ("busy" in q or "busiest" in q or "season" in q):
        return _busy_months_answer(df, target_col)

    # Weekends vs weekdays
    if "weekend" in q or "weekday" in q:
        return _weekend_vs_weekday_answer(df, target_col)

    # Holidays vs non-holidays
    if "holiday" in q:
        return _holiday_answer(df, target_col)

    return None


# -------------------------------------------------------------------
# 4. "Answer" builder using retrieved context (Option A formatting)
# -------------------------------------------------------------------


def _build_answer_from_context(query: str, context_docs: list[str]) -> str:
    """
    Build a simple human-readable answer string from the retrieved documents.
    This is NOT a true LLM, but it's enough to show RAG-style behaviour
    without any external API.
    """
    if not context_docs:
        return (
            "I could not find any rows in the dataset that match your question well. "
            "Try asking in simpler words (for example: 'Which months are busy?' "
            "or 'Do weekends have more visitors than weekdays?')."
        )

    bullets = "\n".join(f"- {doc}" for doc in context_docs)

    answer = (
        f"You asked:\n\n"
        f"**{query}**\n\n"
        "Here are some relevant example days from the Banff dataset:\n\n"
        f"{bullets}\n\n"
        "From these examples, you can see how factors like month, weekend/holiday status, "
        "and recent visitor trends relate to the number of visitors on those days."
    )

    return answer


# -------------------------------------------------------------------
# 5. Public function used by app.py
# -------------------------------------------------------------------


def rag_answer(query: str, df: pd.DataFrame) -> str:
    """
    Main entry point called by app.py.

    - Takes the user's question and the Banff dataframe
    - For some common question types, computes a small direct
      data-based summary (Option B).
    - Retrieves similar rows using TF-IDF and turns them into
      natural-language examples (Option A).
    """
    if not isinstance(query, str) or not query.strip():
        return "Please type a non-empty question about the Banff visitor data."

    query = query.strip()

    # 1) Try to build a small analytics-based answer, if we can
    stats_summary = _try_simple_analytics_answer(query, df)

    # 2) Retrieve context rows
    context_docs = retrieve_context(query, df, top_k=5)
    context_text = _build_answer_from_context(query, context_docs)

    # 3) Combine
    if stats_summary:
        combined = (
            "### Short data-based summary\n\n"
            f"{stats_summary}\n\n"
            "---\n\n"
            "### Example days from the dataset\n\n"
            f"{context_text}"
        )
    else:
        combined = context_text

    return combined
