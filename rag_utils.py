from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1. Convert each row of the Banff dataframe into a short text description
# -------------------------------------------------------------------


def _month_to_name(m) -> str:
    """Convert numeric month (1–12) to a short name, if possible."""
    try:
        m_int = int(m)
    except (TypeError, ValueError):
        return str(m)

    names = {
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
    return names.get(m_int, str(m_int))


def _dow_to_name(d) -> str:
    """Convert numeric day_of_week (0–6) to name, if possible."""
    try:
        d_int = int(d)
    except (TypeError, ValueError):
        return str(d)

    names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    return names.get(d_int, str(d_int))


def _flag_to_text(x, true_word="yes", false_word="no") -> str:
    """Convert 0/1, True/False, etc. into simple words."""
    if pd.isna(x):
        return "unknown"

    if isinstance(x, (int, float)):
        return true_word if x >= 0.5 else false_word

    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes"}:
            return true_word
        if s in {"0", "false", "no"}:
            return false_word

    return str(x)


def _row_to_text(row: pd.Series, idx: int) -> str:
    """
    Turn a single row into a short natural-language description.
    We handle missing columns safely with .get().
    """
    # Basic calendar info
    date = row.get("date", None)
    month_raw = row.get("month", "unknown")
    month_name = _month_to_name(month_raw)

    dow_raw = row.get("day_of_week", "unknown")
    dow_name = _dow_to_name(dow_raw) if dow_raw != "unknown" else "unknown weekday"

    # Flags
    is_weekend = _flag_to_text(row.get("is_weekend", "unknown"), "weekend", "weekday")
    is_holiday = _flag_to_text(row.get("is_holiday", "unknown"), "holiday", "non-holiday")

    # Targets / features
    visitors = row.get("daily_visits.1", row.get("daily_visits", "unknown"))
    rolling_7 = row.get("rolling_7", "unknown")
    lag_1 = row.get("lag_1", "unknown")
    lag_7 = row.get("lag_7", "unknown")

    if date is None or pd.isna(date):
        date_part = f"a day in {month_name}"
    else:
        date_part = f"{date} (in {month_name})"

    return (
        f"Row {idx}: On {date_part}, which was a {dow_name} "
        f"({is_weekend}, {is_holiday}), the visitors were {visitors}. "
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
    results: list[str] = []

    for i in top_idx:
        # Skip if similarity is zero (no overlap)
        if sims[i] <= 0:
            continue
        results.append(_DOCS_CACHE[i])

    return results


# -------------------------------------------------------------------
# 3. "Answer" builder (no external LLM, just summarises context)
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
        "You asked:\n\n"
        f"**{query}**\n\n"
        "Here are some relevant example days from the Banff dataset:\n\n"
        f"{bullets}\n\n"
        "From these examples, you can see how factors like month, "
        "weekend/holiday status, and recent visitor trends relate "
        "to the number of visitors on those days."
    )

    return answer


# -------------------------------------------------------------------
# 4. Public function used by app.py
# -------------------------------------------------------------------


def rag_answer(query: str, df: pd.DataFrame) -> str:
    """
    Main entry point called by app.py.

    - Takes the user's question and the Banff dataframe
    - Retrieves similar rows using TF-IDF
    - Returns a simple text answer built from those rows
    """
    if not isinstance(query, str) or not query.strip():
        return "Please type a non-empty question about the Banff visitor data."

    context_docs = retrieve_context(query, df, top_k=5)
    answer = _build_answer_from_context(query, context_docs)
    return answer
