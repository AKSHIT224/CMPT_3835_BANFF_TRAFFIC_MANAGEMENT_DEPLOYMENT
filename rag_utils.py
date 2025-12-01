import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import streamlit as st


# ---------------------------------------------------------
# 0. Local LLM setup (Flan-T5 small by default)
# ---------------------------------------------------------
@st.cache_resource
def get_local_llm():
    """
    Load a small local language model using transformers.
    Default: google/flan-t5-small (good enough for short answers).
    No API keys or external billing needed.

    We cache it so the model is loaded only once per session.
    The first call may take longer while the model downloads.
    """

    model_name = os.getenv("LOCAL_LLM_NAME", "google/flan-t5-small")

    text2text_pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
    )
    return text2text_pipe


# ---------------------------------------------------------
# 1. Build documents from your Banff dataset
# ---------------------------------------------------------
def build_documents_from_banff(df: pd.DataFrame):
    """
    Turn the Banff features into readable text so the model can use them.
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
# 2. Build embeddings + local LLM
# ---------------------------------------------------------
def build_rag_components(df: pd.DataFrame):
    """
    Build documents from df, create their embeddings, and set up local LLM.
    Returns (documents, doc_embeddings, embedder, llm_pipe).
    """
    documents = build_documents_from_banff(df)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    doc_embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }

    llm_pipe = get_local_llm()
    return documents, doc_embeddings, embedder, llm_pipe


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
# 4. LLM call (local Flan-T5)
# ---------------------------------------------------------
def query_llm(query: str, context: str, llm_pipe):
    """
    Construct a prompt with context and query, and call the local T5 model.
    """
    prompt = (
        "You are an assistant helping to analyze Banff visitor and traffic data. "
        "Use ONLY the context below to answer the user's question clearly and simply.\n\n"
        "Summarize patterns and relationships instead of just repeating raw numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}\n\n"
        "Answer in 3–5 short sentences."
    )

    # text2text-generation pipeline returns a list of dicts with 'generated_text'
    result = llm_pipe(
        prompt,
        max_new_tokens=128,
        do_sample=False,
    )
    text = result[0].get("generated_text", "").strip()
    return text


# ---------------------------------------------------------
# 5. RAG chatbot helper
# ---------------------------------------------------------
def rag_answer(query: str, df: pd.DataFrame):
    """
    Convenience function: build components, retrieve context, and answer.
    For a small number of docs this is OK to recompute each question.
    """
    documents, doc_embeddings, embedder, llm_pipe = build_rag_components(df)
    top_doc_ids = retrieve_context_ids(query, embedder, doc_embeddings, top_k=2)
    context = build_context_text(top_doc_ids, documents)
    answer = query_llm(query, context, llm_pipe)
    return answer
