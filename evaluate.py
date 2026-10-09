"""Run RAG evaluation for the MirAI policy advisor."""

import csv
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend import CHAT_MODEL, DEFAULT_HANDBOOK_PATH, answer_question, get_llm, ingest_pdf_document, vectorstore_ready

load_dotenv()

QUESTIONS = [
    "I have 72% attendance. How many attendance marks will I get?",
    "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
    "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
    "How much is the fine for smoking a cigarette on campus?",
]

CSV_PATH = Path("rag_eval_scores.csv")
FIELDNAMES = ["question", "generated_answer", "retrieved_context", "score", "reasoning", "evaluation_status"]

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict RAG evaluator. Grade the answer against only the supplied retrieved context. "
            "Start with exactly `SCORE: N` where N is an integer from 1 to 5, then provide brief reasoning. "
            "Judge factual correctness, source support, completeness, and hallucination avoidance.",
        ),
        (
            "human",
            "Question:\n{question}\n\nRetrieved context:\n{context}\n\nGenerated answer:\n{answer}\n\nEvaluation:",
        ),
    ]
)


def write_rows(rows: List[Dict[str, str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def pending_rows(reason: str) -> List[Dict[str, str]]:
    return [
        {
            "question": q,
            "generated_answer": "",
            "retrieved_context": "",
            "score": "",
            "reasoning": reason,
            "evaluation_status": "pending",
        }
        for q in QUESTIONS
    ]


def main() -> None:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your_google_api_key_here":
        write_rows(pending_rows("GOOGLE_API_KEY is missing; evaluation requires live Gemini access."))
        print(f"Wrote pending evaluation rows to {CSV_PATH}")
        return

    try:
        if not vectorstore_ready():
            ingest_pdf_document(str(DEFAULT_HANDBOOK_PATH), DEFAULT_HANDBOOK_PATH.name)
    except Exception as exc:
        write_rows(pending_rows(f"Ingestion failed: {exc}"))
        print(f"Wrote failed ingestion status to {CSV_PATH}")
        return

    judge = JUDGE_PROMPT | get_llm() | StrOutputParser()
    rows = []
    for question in QUESTIONS:
        try:
            rag = answer_question(question)
            evaluation = judge.invoke(
                {
                    "question": question,
                    "context": rag["retrieved_context"],
                    "answer": rag["answer"],
                }
            )
            score_match = re.search(r"SCORE:\s*([1-5])", evaluation, re.IGNORECASE)
            rows.append(
                {
                    "question": question,
                    "generated_answer": rag["answer"],
                    "retrieved_context": rag["retrieved_context"],
                    "score": score_match.group(1) if score_match else "",
                    "reasoning": evaluation,
                    "evaluation_status": "completed",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "question": question,
                    "generated_answer": "",
                    "retrieved_context": "",
                    "score": "",
                    "reasoning": str(exc),
                    "evaluation_status": "failed",
                }
            )
    write_rows(rows)
    print(f"Wrote evaluation results to {CSV_PATH}")


if __name__ == "__main__":
    main()
