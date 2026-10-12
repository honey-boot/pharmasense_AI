"""
PharmaSense AI — core pipeline
Multi-agent routing, read-only SQL, literature retrieval, and adverse-event triage.

Notes:
- Uses synthetic/project data; not for clinical decision-making.
- Set ANTHROPIC_API_KEY in your .env file.
- Run Streamlit from the project root so the Data/ paths resolve.
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Configuration and LLM setup
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"
DATABASE_PATH = PROJECT_DIR / "pharmasense.db"
ESCALATIONS_PATH = PROJECT_DIR / "escalations.csv"

load_dotenv(PROJECT_DIR / ".env")

MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = Anthropic(api_key=API_KEY) if API_KEY else None

llm_usage: dict[str, int | float] = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "errors": 0,
    "latency_seconds": 0.0,
}


def call_llm(prompt: str, max_tokens: int = 800) -> str:
    """Call Anthropic and record latency/token usage; errors are propagated."""
    if client is None:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to the project's .env file."
        )

    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        llm_usage["calls"] += 1
        llm_usage["input_tokens"] += response.usage.input_tokens
        llm_usage["output_tokens"] += response.usage.output_tokens

        answer = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

        if not answer:
            raise ValueError("The model returned an empty response.")
        return answer
    except Exception:
        llm_usage["errors"] += 1
        raise
    finally:
        llm_usage["latency_seconds"] += time.perf_counter() - start


def get_usage_snapshot() -> dict[str, int | float]:
    """Return a copy of current in-process usage counters."""
    return dict(llm_usage)


# ---------------------------------------------------------------------------
# Data loading and SQLite initialization
# ---------------------------------------------------------------------------

def _read_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            "Check that the Data folder and CSV files are present."
        )
    return pd.read_csv(path)


compounds = _read_csv("compounds.csv")
clinical_trials = _read_csv("clinical_trials.csv")
trial_sites = _read_csv("trial_sites.csv")
lab_results = _read_csv("lab_results.csv")
adverse_events = _read_csv("adverse_events.csv")
research_documents = _read_csv("research_documents.csv")


def initialize_database() -> None:
    """Create/refresh project tables in the local SQLite database."""
    with sqlite3.connect(DATABASE_PATH, timeout=30) as db:
        compounds.to_sql("compounds", db, if_exists="replace", index=False)
        clinical_trials.to_sql("clinical_trials", db, if_exists="replace", index=False)
        trial_sites.to_sql("trial_sites", db, if_exists="replace", index=False)
        lab_results.to_sql("lab_results", db, if_exists="replace", index=False)
        adverse_events.to_sql("adverse_events", db, if_exists="replace", index=False)
        research_documents.to_sql(
            "research_documents", db, if_exists="replace", index=False
        )


initialize_database()


# ---------------------------------------------------------------------------
# Literature retrieval (TF-IDF RAG)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    words = str(text).split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


all_chunks: list[dict[str, Any]] = []
for _, row in research_documents.iterrows():
    for chunk in chunk_text(row.get("full_text", "")):
        all_chunks.append(
            {
                "doc_id": row.get("doc_id", "UNKNOWN"),
                "title": row.get("title", "Untitled document"),
                "compound_id": row.get("compound_id"),
                "trial_id": row.get("trial_id"),
                "chunk_text": chunk,
            }
        )

if all_chunks:
    chunk_texts = [item["chunk_text"] for item in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
else:
    vectorizer = None
    tfidf_matrix = None


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return relevant TF-IDF chunks; score is a retrieval signal, not truth."""
    if not query or not query.strip() or vectorizer is None or tfidf_matrix is None:
        return []

    try:
        query_vector = vectorizer.transform([query])
    except ValueError:
        # For example, the query contains no terms in the fitted vocabulary.
        return []

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][: max(1, top_k)]
    results: list[dict[str, Any]] = []

    for index in top_indices:
        if scores[index] <= 0:
            continue
        item = all_chunks[index]
        results.append(
            {
                "doc_id": item["doc_id"],
                "title": item["title"],
                "compound_id": item["compound_id"],
                "trial_id": item["trial_id"],
                "score": float(scores[index]),
                "chunk_text": item["chunk_text"],
            }
        )
    return results


def citation_formatter_tool(results: list[dict[str, Any]]) -> str:
    """Format unique document citations."""
    seen: set[str] = set()
    citations: list[str] = []
    for result in results:
        citation = f"[{result['doc_id']}] {result['title']}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return "\n".join(citations)


def literature_research_agent(question: str) -> str:
    results = search(question, top_k=3)
    if not results:
        return (
            "GUARDRAIL: I couldn't find relevant supporting evidence in the "
            "available research documents. Please try a more specific question "
            "or provide a relevant source."
        )

    # This is a simple lexical-retrieval guardrail. Tune using a reviewed test set.
    min_retrieval_score = float(os.getenv("RAG_MIN_SCORE", "0.01"))
    if max(result["score"] for result in results) < min_retrieval_score:
        return (
            "GUARDRAIL: The available documents did not provide sufficiently "
            "relevant evidence to answer this question."
        )

    context = "\n\n".join(
        f"Document [{item['doc_id']}] — {item['title']}\n{item['chunk_text']}"
        for item in results
    )
    citations = citation_formatter_tool(results)
    prompt = f"""You are a research assistant for pharmaceutical R&D.
Answer the question using ONLY the supplied passages.
If the passages do not contain the answer, say that the available evidence is
insufficient. Do not invent facts, results, citations, or clinical advice.
Distinguish reported evidence from uncertainty. Keep the answer concise.

Question:
{question}

Retrieved passages:
{context}

Return an answer supported by the passages."""
    try:
        answer = call_llm(prompt)
    except Exception:
        return (
            "FALLBACK: The literature research service is temporarily unavailable. "
            "No verified answer was generated. Please try again later."
        )

    return f"{answer}\n\nSources:\n{citations}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def sql_query_tool(query: str) -> dict[str, Any]:
    """Execute a single read-only SELECT/WITH query against project SQLite."""
    cleaned = query.strip().rstrip(";")
    if not cleaned:
        return {"error": "Empty SQL query."}

    # SQLite's query_only prevents write operations on this connection.
    # Restrict accepted statements to SELECT or WITH and one statement.
    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", cleaned):
        return {"error": "Only read-only SELECT queries are allowed."}
    if ";" in cleaned:
        return {"error": "Only one SQL statement is allowed."}

    try:
        with sqlite3.connect(
            f"file:{DATABASE_PATH.as_posix()}?mode=ro", uri=True, timeout=10
        ) as db:
            db.execute("PRAGMA query_only = ON")
            result_df = pd.read_sql_query(cleaned, db)
        return {
            "rows": result_df.to_dict(orient="records"),
            "row_count": int(len(result_df)),
        }
    except Exception:
        # Avoid returning database internals or raw SQL errors to the UI.
        return {"error": "The read-only database query could not be completed."}


def compound_similarity_tool(
    compound_id: str, top_n: int = 5
) -> dict[str, Any]:
    target = compounds[compounds["compound_id"].astype(str) == str(compound_id)]
    if target.empty:
        return {"error": f"compound_id {compound_id} not found"}

    target_row = target.iloc[0]
    target_area = target_row.get("therapeutic_area")
    target_mechanism = target_row.get("mechanism_of_action")

    same_area = compounds["therapeutic_area"].eq(target_area)
    same_mechanism = compounds["mechanism_of_action"].eq(target_mechanism)
    similar = compounds[
        (compounds["compound_id"].astype(str) != str(compound_id))
        & (same_area | same_mechanism)
    ].head(max(1, top_n))

    return {
        "target": target_row[
            ["compound_id", "compound_name", "therapeutic_area", "mechanism_of_action"]
        ].to_dict(),
        "similar_compounds": similar[
            ["compound_id", "compound_name", "therapeutic_area", "mechanism_of_action"]
        ].to_dict(orient="records"),
    }


def escalation_notifier_tool(event_id: str, reason: str) -> dict[str, str]:
    """Append a local escalation record; this does not send an external alert."""
    ESCALATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ESCALATIONS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [time.strftime("%Y-%m-%d %H:%M:%S"), event_id, reason]
        )
    return {"status": "escalated", "event_id": event_id}


def ae_severity_classifier_tool(event_description: str) -> str:
    prompt = f"""You are assisting a qualified pharmacovigilance reviewer.
This is a preliminary triage of a synthetic/project adverse-event report, not a
clinical diagnosis. Do not infer seriousness or causality from timing alone.
If information is insufficient, explicitly state that and recommend human review.

Event description:
{event_description}

Respond in exactly this format:
Severity: <Mild, Moderate, Severe, or Undetermined>
Escalate: <Yes or No>
Reason: <one concise sentence>
Missing information: <key missing details, or None>"""
    return call_llm(prompt, max_tokens=300)


# ---------------------------------------------------------------------------
# Agents and orchestration
# ---------------------------------------------------------------------------

def clean_sql(raw_sql: str) -> str:
    cleaned = raw_sql.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def ae_triage_agent(event_id: str, event_description: str) -> str:
    if not event_description or not event_description.strip():
        return (
            "TRIAGE STATUS: INCOMPLETE\n"
            "No event description was provided. Human review is required."
        )

    try:
        classification = ae_severity_classifier_tool(event_description)
    except Exception:
        return (
            "TRIAGE STATUS: HUMAN REVIEW REQUIRED\n"
            "The triage service could not complete its assessment. "
            "No verified classification was generated."
        )

    text = classification.lower()
    severity_match = re.search(
        r"(?im)^\s*severity:\s*(mild|moderate|severe|undetermined)\s*$",
        classification,
    )
    escalation_match = re.search(
        r"(?im)^\s*escalate:\s*(yes|no)\s*$", classification
    )

    if not severity_match or not escalation_match:
        return (
            "TRIAGE STATUS: HUMAN REVIEW REQUIRED\n"
            "The model returned an invalid or incomplete format. "
            "No automated classification decision was accepted."
        )

    severity = severity_match.group(1).lower()
    escalate = escalation_match.group(1).lower() == "yes"

    # Conservative handling: uncertain severity must be reviewed.
    if severity in {"severe", "undetermined"}:
        escalate = True

    if escalate:
        try:
            escalation_notifier_tool(
                event_id,
                f"Preliminary AI triage flagged for review; severity={severity}",
            )
        except Exception:
            return (
                f"{classification}\n\n"
                "STATUS: HUMAN REVIEW REQUIRED. The local escalation record "
                "could not be written; notify the responsible reviewer "
                "through the approved process."
            )

        status = "ESCALATED TO HUMAN REVIEWER"
    else:
        status = (
            "PRELIMINARY TRIAGE ONLY — follow the approved review process"
        )

    return f"{classification}\n\n[STATUS: {status}]"


def classify_intent(question: str) -> str:
    prompt = f"""Classify the user's question into exactly one category.
Return ONLY one category word from this list:
sql
rag
ae_triage
similarity
full_picture
unknown

sql: trial status, enrollment, phases, sites, or lab values
rag: literature, research findings, or research documents
ae_triage: an adverse event report needing preliminary triage
similarity: compounds similar to another compound
full_picture: a complete summary spanning trials and literature for one compound
unknown: anything else or unclear intent

User question:
{question}

Category:"""

    result = call_llm(prompt).strip().lower()
    match = re.search(r"\b(sql|rag|ae_triage|similarity|full_picture|unknown)\b", result)
    return match.group(1) if match else "unknown"


def full_picture_agent(question: str, compound_id: str) -> str:
    compound_rows = compounds[
        compounds["compound_id"].astype(str) == str(compound_id)
    ]
    if compound_rows.empty:
        return f"Compound {compound_id} was not found in the dataset."

    compound_name = str(compound_rows.iloc[0]["compound_name"])
    safe_id = compound_id.replace("'", "''")

    sql = f"""
        SELECT trial_id, compound_id, trial_phase, status,
               actual_enrollment, target_enrollment
        FROM clinical_trials
        WHERE compound_id = '{safe_id}'
    """
    trial_data = sql_query_tool(sql)
    if "error" in trial_data:
        return "I couldn't retrieve the trial data from the read-only database."

    literature_data = literature_research_agent(
        f"What research exists about {compound_name}?"
    )

    prompt = f"""Prepare a concise report about compound {compound_id}
({compound_name}) using ONLY the supplied data. Clearly distinguish the
trial-table data from literature findings. Do not invent missing information.

TRIAL DATA:
{trial_data}

LITERATURE FINDINGS:
{literature_data}

User's question:
{question}"""
    try:
        return call_llm(prompt)
    except Exception:
        return (
            "FALLBACK: The full-picture report could not be generated because "
            "the AI service failed. Please try again later."
        )


def master_router(question: str) -> str:
    """Route a question to the relevant agent and return a user-safe response."""
    if not question or not question.strip():
        return "Please enter a question about trials, compounds, safety, or literature."

    try:
        intent = classify_intent(question)
    except Exception:
        return (
            "FALLBACK: I couldn't classify your question because the AI service "
            "is unavailable. Please try again later."
        )

    try:
        if intent == "sql":
            sql_prompt = f"""Write one read-only SQLite SELECT query for this question.
Use only this table and these columns:
clinical_trials(trial_id, compound_id, trial_phase, therapeutic_area,
sponsor, start_date, planned_end_date, actual_end_date, status,
target_enrollment, actual_enrollment, primary_endpoint)

Question: {question}

Return only the SQL query. Do not use multiple statements."""
            sql = clean_sql(call_llm(sql_prompt))
            result = sql_query_tool(sql)
            if "error" in result:
                return (
                    "I couldn't safely complete that database query. "
                    "Please rephrase the question."
                )
            return f"Query used:\n{sql}\n\nResult:\n{result}"

        if intent == "rag":
            return literature_research_agent(question)

        if intent == "ae_triage":
            event_id_match = re.search(r"\bAE[-_]\w+\b", question, flags=re.IGNORECASE)
            event_id = event_id_match.group(0) if event_id_match else "UNSPECIFIED"
            return ae_triage_agent(event_id, question)

        if intent == "similarity":
            match = re.search(r"CMP-\d{4}", question, flags=re.IGNORECASE)
            if not match:
                return "Please specify a compound_id such as CMP-0001."
            return str(compound_similarity_tool(match.group().upper()))

        if intent == "full_picture":
            match = re.search(r"CMP-\d{4}", question, flags=re.IGNORECASE)
            if not match:
                return "Please specify a compound_id such as CMP-0001."
            return full_picture_agent(question, match.group().upper())

        return (
            "I couldn't identify the right PharmaSense AI agent for that request. "
            "Please ask about clinical trials, research literature, adverse "
            "events, or a compound ID."
        )

    except Exception:
        return (
            "FALLBACK: Something went wrong while processing your request. "
            "No verified answer was generated. Please try again later."
        )


# ---------------------------------------------------------------------------
# Lightweight evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_latency(question: str) -> dict[str, Any]:
    """Measure end-to-end router latency for one test question."""
    start = time.perf_counter()
    answer = master_router(question)
    elapsed = time.perf_counter() - start
    return {
        "question": question,
        "answer": answer,
        "latency_seconds": round(elapsed, 4),
        "llm_usage": get_usage_snapshot(),
    }


def estimate_api_cost(
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Estimate cost using caller-supplied model pricing, in USD."""
    input_cost = (
        int(llm_usage["input_tokens"]) / 1_000_000
    ) * input_price_per_million
    output_cost = (
        int(llm_usage["output_tokens"]) / 1_000_000
    ) * output_price_per_million
    return round(input_cost + output_cost, 8)
