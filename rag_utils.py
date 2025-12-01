from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1. Convert each row of the Banff dataframe into a short text description
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
        f"Row {idx}: On {date} (month={month}, day_of_week={dow}, "
        f"is_weekend={is_weekend}, is_holiday={is_holiday}), "
        f"the visitors were {visitors}. "
        f"Rolling 7-day average was {rolling_7}, "
        f"lag_1 was {lag_1}, lag_7 was {lag_7}."
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
    results = []

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
        f"You asked:\n\n"
        f"**{query}**\n\n"
        "Here are some relevant example days from the Banff dataset:\n\n"
        f"{bullets}\n\n"
        "From these examples, you can see how factors like month, weekend/holiday, "
        "and recent visitor trends relate to the number of visitors on those days."
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
