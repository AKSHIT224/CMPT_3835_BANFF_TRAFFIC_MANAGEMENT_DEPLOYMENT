import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import streamlit as st


# ---------------------------------------------------------
# 0. Gemini setup
# ---------------------------------------------------------
def get_gemini_model():
    """
    Configure and return a Gemini model.
    Tries Streamlit secrets first, then environment variable.
    """
    api_key = None

    # 1) Try Streamlit secrets (for Streamlit Cloud)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets may not exist outside Streamlit
        pass

    # 2) Fallback to environment variable (for local testing)
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Set it in Streamlit secrets (for cloud) or as an environment variable "
            "GEMINI_API_KEY (for local testing)."
        )

    genai.configure(api_key=api_key)
    # Use flash (fast) or pro (stronger)
    return genai.GenerativeModel("gemini-1.5-flash")


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
    """
    prompt = (
        "You are an assistant helping to analyze Banff visitor and traffic data. "
        "Use ONLY the context below to answer the user's question clearly and simply.\n\n"
        "Summarize patterns and relationships instead of just repeating raw numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}\n\n"
        "Answer in 3–5 sentences:"
    )

    response = model.generate_content(prompt)
    return (response.text or "").strip()


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
    
