# pipeline.py
# Idhu namma notebook-la build panna ella functions-um oru file-a sernthu irukku

import pandas as pd
import sqlite3
import os
import re
import csv
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Step 2: LLM setup ----------------
load_dotenv()
api_key = os.environ.get('ANTHROPIC_API_KEY')
client = Anthropic(api_key=api_key)

def call_llm(prompt):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ---------------- Step 1: Data loading ----------------
compounds = pd.read_csv("Data/compounds.csv")
clinical_trials = pd.read_csv("Data/clinical_trials.csv")
trial_sites = pd.read_csv("Data/trial_sites.csv")
lab_results = pd.read_csv("Data/lab_results.csv")
adverse_events = pd.read_csv("Data/adverse_events.csv")
research_documents = pd.read_csv("Data/research_documents.csv")

conn = sqlite3.connect("pharmasense.db")
compounds.to_sql("compounds", conn, if_exists="replace", index=False)
clinical_trials.to_sql("clinical_trials", conn, if_exists="replace", index=False)
trial_sites.to_sql("trial_sites", conn, if_exists="replace", index=False)
lab_results.to_sql("lab_results", conn, if_exists="replace", index=False)
adverse_events.to_sql("adverse_events", conn, if_exists="replace", index=False)
research_documents.to_sql("research_documents", conn, if_exists="replace", index=False)

# ---------------- Step 3: RAG ----------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

all_chunks = []
for idx, row in research_documents.iterrows():
    for chunk in chunk_text(str(row['full_text'])):
        all_chunks.append({
            "doc_id": row['doc_id'], "title": row['title'],
            "compound_id": row['compound_id'], "trial_id": row['trial_id'],
            "chunk_text": chunk
        })

chunk_texts = [c['chunk_text'] for c in all_chunks]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(chunk_texts)

def search(query, top_k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for i in top_indices:
        if scores[i] <= 0:
            continue
        results.append({
            "doc_id": all_chunks[i]['doc_id'], "title": all_chunks[i]['title'],
            "score": scores[i], "chunk_text": all_chunks[i]['chunk_text']
        })
    return results

def citation_formatter_tool(results):
    return "\n".join([f"[{r['doc_id']}] {r['title']}" for r in results])

# ---------------- Step 5: Tools ----------------
def sql_query_tool(query):
    blocked_words = ["insert", "update", "delete", "drop", "alter"]
    if any(word in query.lower() for word in blocked_words):
        return {"error": "Only SELECT queries are allowed."}
    try:
        # ovvoru query-kum PUTHU connection create pannunga (thread-safe)
        local_conn = sqlite3.connect("pharmasense.db")
        result_df = pd.read_sql_query(query, local_conn)
        local_conn.close()
        return {"rows": result_df.to_dict(orient="records"), "row_count": len(result_df)}
    except Exception as e:
        return {"error": str(e)}

def compound_similarity_tool(compound_id, top_n=5):
    target = compounds[compounds['compound_id'] == compound_id]
    if target.empty:
        return {"error": f"compound_id {compound_id} not found"}
    target_area = target.iloc[0]['therapeutic_area']
    target_mechanism = target.iloc[0]['mechanism_of_action']
    similar = compounds[
        (compounds['compound_id'] != compound_id) &
        ((compounds['therapeutic_area'] == target_area) |
         (compounds['mechanism_of_action'] == target_mechanism))
    ].head(top_n)
    return {
        "target": target.iloc[0][['compound_id', 'compound_name', 'therapeutic_area', 'mechanism_of_action']].to_dict(),
        "similar_compounds": similar[['compound_id', 'compound_name', 'therapeutic_area', 'mechanism_of_action']].to_dict(orient="records")
    }

def escalation_notifier_tool(event_id, reason):
    with open("escalations.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event_id, reason])
    return {"status": "escalated", "event_id": event_id}

def ae_severity_classifier_tool(event_description):
    prompt = f"""You are an Adverse Event Triage specialist.
Classify this adverse event report.

Event: "{event_description}"

Respond in EXACTLY this format:
Severity: <Mild, Moderate, or Severe>
Escalate: <Yes or No>
Reason: <one sentence>"""
    return call_llm(prompt)

# ---------------- Step 6: Agents + Orchestration ----------------
def clean_sql(raw_sql):
    cleaned = raw_sql.strip()
    cleaned = cleaned.replace("```sql", "").replace("```", "")
    return cleaned.strip()

def literature_research_agent(question):
    results = search(question, top_k=3)
    if not results:
        return "I don't have internal research on this topic."
    context = "\n\n".join([r['chunk_text'] for r in results])
    citations = citation_formatter_tool(results)
    prompt = f"""Answer the question using ONLY the information in the passages below.
Do not use any outside knowledge.

Question: {question}

Passages:
{context}

Answer in 2-3 sentences."""
    answer = call_llm(prompt)
    return f"{answer}\n\nSources:\n{citations}"

def ae_triage_agent(event_id, event_description):
    classification = ae_severity_classifier_tool(event_description)
    if "escalate: yes" in classification.lower():
        escalation_notifier_tool(event_id, "Auto-escalated per triage rules")
        return f"{classification}\n\n[STATUS: ESCALATED to human reviewer]"
    return f"{classification}\n\n[STATUS: Not escalated, routine monitoring]"

def classify_intent(question):
    prompt = f"""Classify this question into EXACTLY ONE category.
Respond with ONLY the category word, nothing else.

Categories:
sql - about trial status, enrollment, phases, sites, lab values
rag - about literature, research findings, documents
ae_triage - reporting or asking about an adverse event needing triage
similarity - asking which compounds are similar to another
full_picture - asking for a full/complete summary spanning trials AND literature for one compound

Question: "{question}"

Category:"""
    result = call_llm(prompt)
    return result.strip().lower()

def full_picture_agent(question, compound_id):
    compound_row = compounds[compounds['compound_id'] == compound_id]
    compound_name = compound_row.iloc[0]['compound_name']

    trial_prompt = f"""Write ONE read-only SQLite SELECT query to find all
clinical trials for compound_id = '{compound_id}'.
Table: clinical_trials(trial_id, compound_id, trial_phase, status,
actual_enrollment, target_enrollment)
Respond with ONLY the SQL query."""
    sql = clean_sql(call_llm(trial_prompt))
    trial_data = sql_query_tool(sql)

    literature_data = literature_research_agent(f"What research exists about {compound_name}?")

    merge_prompt = f"""Merge these two findings into ONE coherent report
about compound {compound_id} ({compound_name}).

TRIAL DATA:
{trial_data}

LITERATURE FINDINGS:
{literature_data}

Write a short, well-organized summary covering both."""
    return call_llm(merge_prompt)

def master_router(question):
    intent = classify_intent(question)

    if intent == "sql":
        sql_prompt = f"""Write ONE read-only SQLite SELECT query for this question.
Table: clinical_trials(trial_id, compound_id, trial_phase, therapeutic_area,
sponsor, start_date, planned_end_date, actual_end_date, status,
target_enrollment, actual_enrollment, primary_endpoint)

Question: {question}
Respond with ONLY the SQL query, no explanation."""
        sql = clean_sql(call_llm(sql_prompt))
        result = sql_query_tool(sql)
        return f"Query used: {sql}\n\nResult: {result}"

    elif intent == "rag":
        return literature_research_agent(question)

    elif intent == "ae_triage":
        return ae_triage_agent("UNSPECIFIED", question)

    elif intent == "similarity":
        match = re.search(r"CMP-\d{4}", question)
        if match:
            return str(compound_similarity_tool(match.group()))
        return "Please specify a compound_id like CMP-0001."

    elif intent == "full_picture":
        match = re.search(r"CMP-\d{4}", question)
        if match:
            return full_picture_agent(question, match.group())
        return "Please specify a compound_id like CMP-0001."

    else:
        return f"Could not classify this question (got: {intent})"