import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------- Optional Supabase client ----------
def get_supabase():
    try:
        from supabase import create_client, Client  # pip install supabase
    except Exception:
        return None
    if "supabase" not in st.secrets:
        return None
    url = st.secrets["supabase"].get("url")
    key = st.secrets["supabase"].get("key")  # Use the service role key on Streamlit Cloud
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

SUPABASE = get_supabase()

# ---------- Paths (for local dev fallback only) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("QUIZ_DATA_DIR", os.path.join(BASE_DIR, "data"))
if not SUPABASE:
    os.makedirs(DATA_DIR, exist_ok=True)
RESPONSES_CSV = os.path.join(DATA_DIR, "responses.csv")   # fallback only
SUMMARIES_CSV = os.path.join(DATA_DIR, "summaries.csv")   # fallback only

QUESTIONS_CSV = os.path.join(BASE_DIR, "questions.csv")

# ---------- Load questions ----------
@st.cache_data
def load_questions(csv_path: str) -> List[Dict]:
    df = pd.read_csv(
        csv_path,
        dtype={"question_id": "string", "question_text": "string",
               "choice_label": "string", "choice_text": "string"},
        keep_default_na=False,
    )
    df.columns = [c.lower() for c in df.columns]
    required = {"question_id", "question_text", "choice_label", "choice_text", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

    df["weight"] = df["weight"].astype(float)
    bad = df[(df["weight"] < 0) | (df["weight"] > 1)]
    if not bad.empty:
        raise RuntimeError("All weights must be within [0,1].")

    questions: List[Dict] = []
    for qid, g in df.groupby("question_id", sort=True):
        q = {
            "id": str(qid),
            "text": str(g["question_text"].iloc[0]),
            "choices": [
                {"label": str(r["choice_label"]).strip(),
                 "text": str(r["choice_text"]).strip(),
                 "weight": float(r["weight"])}
                for _, r in g.iterrows()
            ],
        }
        q["choices"].sort(key=lambda c: c["label"])
        questions.append(q)
    return questions

def compute_scores(questions: List[Dict], answers: Dict[str, str]):
    per_question = []
    for q in questions:
        chosen = answers.get(q["id"])
        if chosen is None:
            raise ValueError(f"Missing answer for question {q['id']}")
        opt = next((c for c in q["choices"] if c["label"] == chosen), None)
        if not opt:
            labels = ", ".join([c["label"] for c in q["choices"]])
            raise ValueError(f"Invalid choice '{chosen}' for {q['id']} (valid: {labels})")
        per_question.append({
            "question_id": q["id"],
            "choice_label": opt["label"],
            "earned": float(opt["weight"]),
            "max": 1.0,
        })
    max_total = float(len(per_question))
    total = float(sum(x["earned"] for x in per_question))
    normalized = total / max_total if max_total > 0 else 0.0
    return per_question, total, max_total, normalized

def persist_submission(student_name: str, per_q: List[Dict],
                       total: float, max_total: float, normalized: float) -> str:
    """Writes to Supabase if configured; else appends to local CSVs."""
    submission_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    detail_rows = [{
        "timestamp": ts,
        "submission_id": submission_id,
        "student_name": student_name,
        "question_id": x["question_id"],
        "choice_label": x["choice_label"],
        "earned": x["earned"],
        "max": x["max"],
        "total_score": total,
        "max_total": max_total,
        "normalized": normalized,
    } for x in per_q]

    summary_row = {
        "timestamp": ts,
        "submission_id": submission_id,
        "student_name": student_name,
        "total_score": total,
        "max_total": max_total,
        "normalized": normalized,
    }

    if SUPABASE:
        # Insert into tables: 'responses' (many rows), 'summaries' (one row)
        try:
            SUPABASE.table("responses").insert(detail_rows).execute()
            SUPABASE.table("summaries").insert(summary_row).execute()
            return submission_id
        except Exception as e:
            # Soft-fallback to CSV (dev-friendly) if Supabase insert fails
            st.warning(f"Supabase insert failed, falling back to CSV: {e}")

    # Fallback: local CSV (OK for local dev; ephemeral on Streamlit Cloud)
    pd.DataFrame(detail_rows).to_csv(
        RESPONSES_CSV, mode="a", index=False, header=not os.path.exists(RESPONSES_CSV)
    )
    pd.DataFrame([summary_row]).to_csv(
        SUMMARIES_CSV, mode="a", index=False, header=not os.path.exists(SUMMARIES_CSV)
    )
    return submission_id

# ---------- UI ----------
st.set_page_config(page_title="Personality Quiz", page_icon="📝", layout="centered")
st.title("Personality Quiz")

try:
    questions = load_questions(QUESTIONS_CSV)
except Exception as e:
    st.error(f"Failed to load questions: {e}")
    st.stop()

name = st.text_input("Student Name", key="student_name", placeholder="e.g. Ayu Permata")

answers: Dict[str, str] = {}
for q in questions:
    st.markdown(f"**{q['id']}. {q['text']}**")
    key = f"q_{q['id']}"
    options = [f"{c['label']}. {c['text']}" for c in q["choices"]]
    selected = st.radio("", options=options, index=None, key=key, label_visibility="collapsed")
    if selected:
        answers[q["id"]] = selected.split(".")[0]  # "A. ..." -> "A"

submitted = st.button("Submit", type="primary", use_container_width=True)

if submitted:
    try:
        if not name.strip():
            st.error("Please enter your name.")
            st.stop()

        missing = [q["id"] for q in questions if q["id"] not in answers]
        if missing:
            st.error(f"Please answer all questions. Missing: {', '.join(missing)}")
            st.stop()

        per_q, total, max_total, normalized = compute_scores(questions, answers)
        submission_id = persist_submission(name.strip(), per_q, total, max_total, normalized)

        st.success(f"Submitted! ID: {submission_id}")

        # Reset UI: clear name + answers
        st.session_state["student_name"] = ""
        for q in questions:
            st.session_state.pop(f"q_{q['id']}", None)
        st.rerun()

    except Exception as e:
        st.error(f"Submit failed: {e}")
