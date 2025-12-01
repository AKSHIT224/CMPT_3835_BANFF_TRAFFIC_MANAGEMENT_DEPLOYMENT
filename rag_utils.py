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


def _bool_flag_to_text(val, weekend=True) -> str:
    """Map 0/1 flags to friendly text."""
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
        return float(val)
    except Exception:
        return "unknown"


def _row_to_text(row: pd.Series, idx: int) -> str:
    """
    Turn a single row into a short natural-language description.
    We handle missing columns safely with .get().
    """
    date = row.get("date", "unknown date")

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
    lag_1 = _safe_number(row.get("lag_1", "unknown"))
    lag_7 = _safe_number(row.get("lag_7", "unknown"))

    return (
        f"Row {idx}: On {date} (a day in {month_txt}, {dow}; "
        f"{weekend_txt}, {holiday_txt}), the visitors were {visitors}. "
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
    Given a free-text query, return up to top_k *distinct* document strings.

    We still rank by similarity using TF-IDF, but we skip near-duplicate
    rows (same text content after 'Row X:') to make the examples
    more varied for the user / presentation.
    """
    _ensure_index(df)

    if not _DOCS_CACHE or _VECTORIZER_CACHE is None or _DOC_MATRIX_CACHE is None:
        return []

    q_vec = _VECTORIZER_CACHE.transform([query])
    sims = cosine_similarity(q_vec, _DOC_MATRIX_CACHE)[0]

    # Sort all indices by similarity (highest first)
    sorted_idx = np.argsort(sims)[::-1]

    results: list[str] = []
    seen_canonical: set[str] = set()

    for i in sorted_idx:
        # If similarity is zero or negative, remaining docs won't help
        if sims[i] <= 0:
            break

        doc = _DOCS_CACHE[i]

        # Build a "canonical" version of the text that ignores the row number.
        # Example: "Row 5: ..." and "Row 3: ..." become the same canonical string.
        if ":" in doc:
            canonical = doc.split(":", 1)[1].strip()
        else:
            canonical = doc.strip()

        # Skip near-duplicates
        if canonical in seen_canonical:
            continue

        seen_canonical.add(canonical)
        results.append(doc)

        if len(results) >= top_k:
            break

    return results


# -------------------------------------------------------------------
# 3. Simple aggregate summaries (months / weekends / holidays)
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


def _build_aggregate_summary(query: str, df: pd.DataFrame) -> str:
    """
    Build a short numeric summary based on the type of question.
    Currently handles:
      - busy months / seasons
      - weekend vs weekday
      - holiday vs non-holiday
    """
    if df is None or df.empty:
        return ""

    q = query.lower()
    value_col = "daily_visits.1"
    if value_col not in df.columns:
        # fallback: try daily_visits
        if "daily_visits" in df.columns:
            value_col = "daily_visits"
        else:
            return ""

    parts: list[str] = []

    # A) By month
    if any(k in q for k in ["month", "season", "busy"]):
        by_month = _safe_group_mean(df, value_col, "month")
        if by_month is not None:
            top = by_month.head(3)
            month_bits = [
                f"{_month_name(m)} (~{v:.0f} visitors/day)" for m, v in top.items()
            ]
            parts.append(
                "The months with the highest average daily visitors are: "
                + ", ".join(month_bits)
                + ". These months look like the busier period in this dataset."
            )

    # B) Weekend vs weekday
    if any(k in q for k in ["weekend", "weekday"]):
        by_weekend = _safe_group_mean(df, value_col, "is_weekend")
        if by_weekend is not None:
            d = by_weekend.to_dict()
            weekend_val = d.get(1, d.get(1.0, None))
            weekday_val = d.get(0, d.get(0.0, None))
            if weekend_val is not None and weekday_val is not None:
                parts.append(
                    f"On average, weekends have about {weekend_val:.0f} visitors/day "
                    f"versus {weekday_val:.0f} visitors/day on weekdays."
                )

    # C) Holiday vs non-holiday
    if "holiday" in q:
        by_holiday = _safe_group_mean(df, value_col, "is_holiday")
        if by_holiday is not None:
            d = by_holiday.to_dict()
            holiday_val = d.get(1, d.get(1.0, None))
            nonholiday_val = d.get(0, d.get(0.0, None))
            if holiday_val is not None and nonholiday_val is not None:
                parts.append(
                    f"Holidays have about {holiday_val:.0f} visitors/day, "
                    f"compared to {nonholiday_val:.0f} on non-holidays."
                )

    if not parts:
        return ""

    return "### Short data-based summary\n\n" + " ".join(parts) + "\n\n"


# -------------------------------------------------------------------
# 4. "Answer" builder (no external LLM, just summaries + examples)
# -------------------------------------------------------------------


def _build_answer_from_context(
    query: str, context_docs: list[str], df: pd.DataFrame
) -> str:
    """
    Build a simple human-readable answer string from:
      - small numeric summary (optional)
      - retrieved example rows
    This is NOT a true LLM, but it is enough to show RAG-style behaviour
    without any external API.
    """
    summary = _build_aggregate_summary(query, df)

    if not context_docs and not summary:
        return (
            "I could not find any rows in the dataset that match your question well. "
            "Try asking in simpler words (for example: 'Which months are busy?' "
            "or 'Do weekends have more visitors than weekdays?')."
        )

    bullets = "\n".join(f"- {doc}" for doc in context_docs)

    answer_parts = []

    if summary:
        answer_parts.append(summary)

    answer_parts.append("### Example days from the dataset\n")
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
    - Computes simple summary stats (months / weekends / holidays) when relevant
    - Retrieves similar rows using TF-IDF, with basic de-duplication for variety
    - Returns a plain-text answer built from those pieces
    """
    if not isinstance(query, str) or not query.strip():
        return "Please type a non-empty question about the Banff visitor data."

    context_docs = retrieve_context(query, df, top_k=5)
    answer = _build_answer_from_context(query, context_docs, df)
    return answer
