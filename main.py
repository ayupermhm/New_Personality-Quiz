from typing import List, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import threading
import os
from datetime import datetime
import uuid

# ---------- Paths ----------
QUESTIONS_CSV = "questions.csv"
RESPONSES_CSV = "responses.csv"     # detailed (one row per question)
SUMMARIES_CSV = "summaries.csv"     # summary (one row per submission)

# ---------- Data models ----------

class Choice(BaseModel):
    label: str = Field(..., description="Short code like 'A'")
    text: str = Field(..., description="Visible choice text")
    weight: float = Field(..., ge=0.0, le=1.0, description="0..1 percentage of the 1-point question")

class Question(BaseModel):
    id: str
    text: str
    choices: List[Choice]

class Answer(BaseModel):
    question_id: str
    choice_label: str

class SubmitRequest(BaseModel):
    student_name: str = Field(..., min_length=1)
    answers: List[Answer]

    @validator("answers")
    def nonempty(cls, v):
        if not v:
            raise ValueError("answers must not be empty")
        return v

class QuestionScore(BaseModel):
    question_id: str
    choice_label: str
    earned: float
    max: float = 1.0

class SubmitResponse(BaseModel):
    submission_id: str
    student_name: str
    per_question: List[QuestionScore]
    total_score: float
    max_total: float
    normalized: float
    saved_rows: int

# ---------- App + middleware ----------

app = FastAPI(
    title="Personality Quiz Backend + UI",
    version="1.2.0",
    description="CSV-driven MCQ quiz with simple UI and CSV logging of submissions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Serve static UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory cache + locks
_QUESTIONS: Dict[str, Question] = {}
_LOAD_LOCK = threading.Lock()
_SAVE_LOCK = threading.Lock()

# ---------- CSV loading ----------

def load_questions(csv_path: str) -> Dict[str, Question]:
    try:
        df = pd.read_csv(
            csv_path,
            dtype={
                "question_id": "string",
                "question_text": "string",
                "choice_label": "string",
                "choice_text": "string",
            },
            keep_default_na=False
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")

    df.columns = [c.lower() for c in df.columns]
    required = {"question_id", "question_text", "choice_label", "choice_text", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

    try:
        df["weight"] = df["weight"].astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed converting 'weight' to float: {e}")

    bad = df[(df["weight"] < 0) | (df["weight"] > 1)]
    if not bad.empty:
        raise RuntimeError("All weights must be within [0,1].")

    grouped: Dict[str, Question] = {}
    for qid, g in df.groupby("question_id", sort=True):
        qtext = str(g["question_text"].iloc[0])
        choices: List[Choice] = []
        for _, row in g.iterrows():
            choices.append(Choice(
                label=str(row["choice_label"]).strip(),
                text=str(row["choice_text"]).strip(),
                weight=float(row["weight"])
            ))
        choices.sort(key=lambda c: c.label)
        grouped[str(qid)] = Question(id=str(qid), text=qtext, choices=choices)

    if not grouped:
        raise RuntimeError("No questions found in CSV.")
    return grouped

def ensure_loaded():
    global _QUESTIONS
    with _LOAD_LOCK:
        if not _QUESTIONS:
            _QUESTIONS = load_questions(QUESTIONS_CSV)

# ---------- Routes ----------

@app.get("/")
def root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "UI not found. Put index.html under ./static. API: /docs, /quiz, /submit, /reload"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/quiz", response_model=List[Question])
def get_quiz(include_weights: bool = Query(False, description="If true, include weights (useful for testing)")):
    ensure_loaded()
    quiz = list(_QUESTIONS.values())
    if include_weights:
        return quiz
    # mask weights by default
    masked = []
    for q in quiz:
        masked.append(Question(
            id=q.id,
            text=q.text,
            choices=[Choice(label=c.label, text=c.text, weight=0.0) for c in q.choices]
        ))
    return masked

@app.post("/submit", response_model=SubmitResponse)
def submit(req: SubmitRequest):
    ensure_loaded()
    per_q: List[QuestionScore] = []

    for ans in req.answers:
        q = _QUESTIONS.get(ans.question_id)
        if not q:
            raise HTTPException(status_code=400, detail=f"Unknown question_id: {ans.question_id}")
        choice = next((c for c in q.choices if c.label == ans.choice_label), None)
        if not choice:
            labels = ", ".join(c.label for c in q.choices)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid choice_label '{ans.choice_label}' for question {ans.question_id}. Valid: {labels}"
            )
        per_q.append(QuestionScore(
            question_id=q.id,
            choice_label=choice.label,
            earned=choice.weight
        ))

    max_total = float(len(per_q))
    total = float(sum(x.earned for x in per_q))
    normalized = total / max_total if max_total > 0 else 0.0

    # Persist: detailed (per-question) + summary (per submission)
    submission_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Detailed rows
    detail_rows = [{
        "timestamp": ts,
        "submission_id": submission_id,
        "student_name": req.student_name,
        "question_id": x.question_id,
        "choice_label": x.choice_label,
        "earned": x.earned,
        "max": 1.0,
        "total_score": total,
        "max_total": max_total,
        "normalized": normalized
    } for x in per_q]

    # Summary row (NEW)
    summary_row = {
        "timestamp": ts,
        "submission_id": submission_id,
        "student_name": req.student_name,
        "total_score": total,
        "max_total": max_total,
        "normalized": normalized
    }

    with _SAVE_LOCK:
        # Append detailed
        detail_df = pd.DataFrame(detail_rows)
        new_detail_file = not os.path.exists(RESPONSES_CSV)
        detail_df.to_csv(RESPONSES_CSV, mode="a", index=False, header=new_detail_file)

        # Append summary
        summary_df = pd.DataFrame([summary_row])
        new_summary_file = not os.path.exists(SUMMARIES_CSV)
        summary_df.to_csv(SUMMARIES_CSV, mode="a", index=False, header=new_summary_file)

    return SubmitResponse(
        submission_id=submission_id,
        student_name=req.student_name,
        per_question=per_q,
        total_score=total,
        max_total=max_total,
        normalized=normalized,
        saved_rows=len(detail_rows)
    )

@app.post("/reload")
def reload_csv():
    global _QUESTIONS
    with _LOAD_LOCK:
        _QUESTIONS = load_questions(QUESTIONS_CSV)
    return {"status": "reloaded", "questions": len(_QUESTIONS)}

# ---------- Dev server ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
