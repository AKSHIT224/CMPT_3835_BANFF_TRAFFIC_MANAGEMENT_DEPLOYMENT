from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 0. Small helpers
# -------------------------------------------------------------------


_MONTH_MAP = {
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


def _to_month_name(value) -> str:
    """Convert numeric month to a nice month name if possible."""
    try:
        m_int = int(value)
        if 1 <= m_int <= 12:
            return _MONTH_MAP[m_int]
    except (TypeError, ValueError):
        pass
    return str(value)


def _flag_to_text(flag, positive: str, negative: str) -> str:
    """
    Turn 0/1 (or 0.0/1.0) into readable text like 'weekend' vs 'weekday'.
    """
    if flag in (1, 1.0, "1", "1.0", True):
        return positive
    if flag in (0, 0.0, "0", "0.0", False):
        return negative
    return "unknown"


# -------------------------------------------------------------------
# 1. Convert each row of the Banff dataframe into a short text description
# -------------------------------------------------------------------


def _row_to_text(row: pd.Series, idx: int) -> str:
    """
    Turn a single row into a short natural-language description.
    We handle missing columns safely with .get().
    """
    # Basic fields
    raw_month = row.get("month", "unknown")
    month_name = _to_month_name(raw_month)

    dow = row.get("day_of_week", "unknown weekday")
    is_weekend = _flag_to_text(row.get("is_weekend", "unknown"),
                               "weekend", "weekday")
    is_holiday = _flag_to_text(row.get("is_holiday", "unknown"),
                               "holiday", "non-holiday")

    visitors = row.get("daily_visits.1", row.get("daily_visits", "unknown"))
    rolling_7 = row.get("rolling_7", "unknown")
    lag_1 = row.get("lag_1", "unknown")
    lag_7 = row.get("lag_7", "unknown")

    return (
        f"Row {idx}: On a day in {month_name}, {dow}; "
        f"{is_weekend}, {is_holiday}, the visitors were {visitors}. "
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
# 2. TF-IDF index + cosine similarity for retrieval
# -------------------------------------------------------------------

# Simple in-memory cache so we don't rebuild vectorizer on every call
_DOCS_CACHE = None
_VECTORIZER_CACHE = None
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
# 3A. Small numeric summary for "busy months"
# -------------------------------------------------------------------


def _month_busy_summary(df: pd.DataFrame,
                        target_col: str = "daily_visits.1",
                        top_k: int = 3) -> str | None:
    """
    Compute a simple 'busy months' summary from the dataset.
    Returns a short text like:
    'April (~4000 visitors/day), July (~3800 visitors/day), ...'
    """
    if target_col not in df.columns or "month" not in df.columns:
        return None

    # Drop NaNs, groupby month, sort by mean visitors
    tmp = (
        df[["month", target_col]]
        .dropna()
        .groupby("month")[target_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
    )

    if tmp.empty:
        return None

    pieces = []
    for m_val, avg_vis in tmp.items():
        m_name = _to_month_name(m_val)
        pieces.append(f"{m_name} (~{avg_vis:.0f} visitors/day)")

    return ", ".join(pieces)


# -------------------------------------------------------------------
# 3B. "Answer" builder (no external LLM, just summarises context)
# -------------------------------------------------------------------


def _build_answer_from_context(
    query: str,
    context_docs: list[str],
    df: pd.DataFrame,
    target_col: str = "daily_visits.1",
) -> str:
    """
    Build a simple human-readable answer string from the retrieved documents.
    This is NOT a true LLM, but it's enough to show RAG-style behaviour
    without any external API.
    """

    # Decide if we should include a month-based summary (Option A)
    q_lower = query.lower()
    wants_months = (
        "month" in q_lower
        or "months" in q_lower
        or "season" in q_lower
        or "busy" in q_lower
    )

    month_summary = None
    if wants_months:
        month_summary = _month_busy_summary(df, target_col=target_col)

    # If no context docs at all
    if not context_docs:
        base_msg = (
            "I could not find any rows in the dataset that match your "
            "question well. Try asking in simpler words, for example:\n\n"
            "- 'Which months are busy?'\n"
            "- 'Do weekends have more visitors than weekdays?'\n"
            "- 'How do holidays change visitor numbers?'\n"
        )
        # If we do have a numeric summary, show that even if context is empty
        if wants_months and month_summary:
            return (
                "### Short data-based summary\n\n"
                f"Based on the dataset, the months with the highest "
                f"average daily visitors are:\n\n"
                f"{month_summary}. These months can be considered the "
                "'busy season' in this dataset.\n\n"
                + base_msg
            )
        return base_msg

    # Option B: build bullet list from retrieved rows
    bullets = "\n".join(f"- {doc}" for doc in context_docs)

    # Combine A (numeric summary) + B (example rows)
    parts = []

    if wants_months and month_summary:
        parts.append(
            "### Short data-based summary\n\n"
            "Based on the dataset, the months with the highest "
            "average daily visitors are:\n\n"
            f"{month_summary}. These months can be considered the "
            "'busy season' in this dataset.\n"
        )

    parts.append("### Example days from the dataset\n")
    parts.append("You asked:\n\n")
    parts.append(f"**{query}**\n\n")
    parts.append("Here are some relevant example days from the Banff dataset:\n\n")
    parts.append(bullets)
    parts.append(
        "\n\nFrom these examples, you can see how factors like month, "
        "weekend/holiday status, and recent visitor trends relate to "
        "the number of visitors on those days."
    )

    return "\n".join(parts)


# -------------------------------------------------------------------
# 4. Public function used by app.py
# -------------------------------------------------------------------


def rag_answer(query: str, df: pd.DataFrame) -> str:
    """
    Main entry point called by app.py.

    - Takes the user's question and the Banff dataframe
    - Retrieves similar rows using TF-IDF
    - Builds a short numeric summary (Option A) when relevant
    - Returns a text answer with both summary + example rows (Option B)
    """
    if not isinstance(query, str) or not query.strip():
        return "Please type a non-empty question about the Banff visitor data."

    context_docs = retrieve_context(query, df, top_k=5)
    answer = _build_answer_from_context(
        query,
        context_docs,
        df,
        target_col="daily_visits.1",  # adjust if your target col name changes
    )
    return answer
