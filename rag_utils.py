import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import streamlit as st
from google.api_core.exceptions import NotFound as GoogleNotFound


# ---------------------------------------------------------
# 0. Helpers for model + API key
# ---------------------------------------------------------
def _get_model_name() -> str:
    """
    Decide which Gemini model name to use.

    Priority:
      1) Streamlit secret GEMINI_MODEL_NAME
      2) Environment variable GEMINI_MODEL_NAME
      3) Default: "gemini-pro" (most widely supported text model)
    """
    model_name = None

    # Try Streamlit secrets
    try:
        if "GEMINI_MODEL_NAME" in st.secrets:
            model_name = st.secrets["GEMINI_MODEL_NAME"]
    except Exception:
        # st.secrets may not exist when running locally
        pass

    # Try environment variable
    if not model_name:
        model_name = os.getenv("GEMINI_MODEL_NAME")

    # Final fallback
    if not model_name:
        model_name = "gemini-pro"

    return model_name


def _get_api_key() -> str:
    """
    Get API key from:
      1) Streamlit secrets GEMINI_API_KEY
      2) Environment variable GEMINI_API_KEY
    """
    api_key = None

    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Set it in Streamlit secrets (for cloud) or as an environment variable "
            "GEMINI_API_KEY (for local testing)."
        )

    return api_key


def get_gemini_model():
    """
    Configure and return a Gemini model using google-generativeai.
    """
    api_key = _get_api_key()
    model_name = _get_model_name()

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


# ---------------------------------------------------------
# 1. Build documents from your Banff dataset
# ---------------------------------------------------------
def build_documents_from_banff(df: pd.DataFrame):
    """
    Turn the Banff features into readable text so the LLM can use them.
    We create:
      - doc1: a general description of the dataset
      - doc2: a narrative built from a sample of rows
    """
    desc = (
        "This dataset contains engineered features for daily visitor and traffic "
        "patterns in Banff. Each row represents a time point with features such as "
        "whether it was a weekend or holiday, the month, recent rolling averages, "
        "and lagged visitor counts. The target is daily_visits.1, which represents "
        "the number of visitors that day."
    )

    narrative = "Here are some example days from the Banff traffic and visitor dataset:\n"
    sample = df.head(50)  # limit to keep context small

    for idx, row in sample.iterrows():
        visitors = row.get("daily_visits.1", row.get("daily_visits", "unknown"))
        month = row.get("month", "unknown")
        is_weekend = row.get("is_weekend", "unknown")
        is_holiday = row.get("is_holiday", "unknown")
        rolling_7 = row.get("rolling_7", "unknown")

        narrative += (
            f"Example {idx}: Month {month}, weekend flag {is_weekend}, "
            f"holiday flag {is_holiday}, rolling 7-day average {rolling_7}, "
            f"and daily visitors (target) {visitors}.\n"
        )

    documents = {
        "doc1": desc,
        "doc2": narrative,
    }
    return documents


# ---------------------------------------------------------
# 2. Build embeddings + Gemini model
# ---------------------------------------------------------
def build_rag_components(df: pd.DataFrame):
    """
    Build documents from df, create their embeddings, and set up Gemini model.
    Returns (documents, doc_embeddings, embedder, gemini_model).
    """
    documents = build_documents_from_banff(df)

    # Sentence-transformers model for embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    doc_embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }

    gemini_model = get_gemini_model()
    return documents, doc_embeddings, embedder, gemini_model


# ---------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------
def retrieve_context_ids(query: str, embedder, doc_embeddings, top_k: int = 2):
    """
    Retrieve the most relevant document IDs for the given query.
    """
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    scores = {}
    for doc_id, emb in doc_embeddings.items():
        score = util.pytorch_cos_sim(query_embedding, emb).item()
        scores[doc_id] = score

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]
    return top_doc_ids


def build_context_text(doc_ids, documents):
    """
    Helper: given doc IDs, join the actual text content.
    """
    return "\n\n".join(documents[d] for d in doc_ids)


# ---------------------------------------------------------
# 4. LLM call (Gemini)
# ---------------------------------------------------------
def query_llm(query: str, context: str, model):
    """
    Construct a prompt with context and query, and call Gemini.
    Handles 'model not found' errors gracefully inside Streamlit.
    """
    model_name = _get_model_name()

    prompt = (
        "You are an assistant helping to analyze Banff visitor and traffic data. "
        "Use ONLY the context below to answer the user's question clearly and simply.\n\n"
        "Summarize patterns and relationships instead of just repeating raw numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}\n\n"
        "Answer in 3–5 sentences:"
    )

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        return text.strip()
    except GoogleNotFound:
        # This is the error you're seeing now: model not found for this API version/key
        msg = (
            f"Gemini model '{model_name}' is not available for your API key / project. "
            "Please check that this model name is correct and supported for your key. "
            "You can change it by setting GEMINI_MODEL_NAME in Streamlit secrets or "
            "as an environment variable. A common safe choice is 'gemini-pro'."
        )
        # Show in the app UI
        try:
            st.error(msg)
        except Exception:
            pass
        return msg
    except Exception as e:
        # Catch-all for any other API errors so the app doesn't crash
        msg = f"Error calling Gemini API: {e}"
        try:
            st.error(msg)
        except Exception:
            pass
        return "There was an error calling the Gemini API. Please try again later."


# ---------------------------------------------------------
# 5. RAG chatbot helper
# ---------------------------------------------------------
def rag_answer(query: str, df: pd.DataFrame):
    """
    Convenience function: build components, retrieve context, and answer.
    For a small number of docs this is OK to recompute each question.
    """
    documents, doc_embeddings, embedder, gemini_model = build_rag_components(df)
    top_doc_ids = retrieve_context_ids(query, embedder, doc_embeddings, top_k=2)
    context = build_context_text(top_doc_ids, documents)
    answer = query_llm(query, context, gemini_model)
    return answer
