"""
Autonomous MirAI Student Policy Advisor - FastAPI RAG backend.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
import chromadb
from fastapi.middleware.cors import CORSMiddleware
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "mirai_policies"
DEFAULT_HANDBOOK_PATH = BASE_DIR / "Mirai_SoT_Policy_Handbook_2026.pdf"
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-3.8-flash"

app = FastAPI(
    title="MirAI Student Policy Advisor API",
    description="Source-grounded RAG API for the MirAI Student Policy Handbook 2026.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
_llm: Optional[ChatGoogleGenerativeAI] = None
_vectorstore: Optional[Chroma] = None
_rag_chain = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_context: str


class IngestResponse(BaseModel):
    success: bool
    filename: str
    page_count: int
    chunk_count: int
    stored_document_count: int
    message: str
    persist_directory: str


def _api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "your_google_api_key_here":
        raise ValueError("GOOGLE_API_KEY is missing. Add it to .env or export it before running the API.")
    return key


def _friendly_google_error(exc: Exception, model_name: str, purpose: str) -> str:
    message = str(exc)
    if "API_KEY" in message or "api key" in message.lower() or "permission" in message.lower():
        return f"{purpose} failed because GOOGLE_API_KEY is missing, invalid, or lacks access: {message}"
    if model_name in message or "404" in message or "not found" in message.lower():
        return f"{purpose} model '{model_name}' failed exactly as configured: {message}"
    return f"{purpose} failed: {message}"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=_api_key())
    return _embeddings


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.0, google_api_key=_api_key())
    return _llm


def get_index_stats() -> Dict[str, Any]:
    chroma_path = Path(PERSIST_DIRECTORY)
    if not chroma_path.exists():
        return {"ready": False, "chunk_count": 0}
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        return {"ready": count > 0, "chunk_count": count}
    except Exception:
        return {"ready": False, "chunk_count": 0}


def vectorstore_ready() -> bool:
    return bool(get_index_stats()["ready"])


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIRECTORY,
        )
    return _vectorstore


def format_docs(docs) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "MirAI handbook")
        page = doc.metadata.get("page_number") or (
            doc.metadata.get("page", 0) + 1 if isinstance(doc.metadata.get("page"), int) else "unknown"
        )
        parts.append(f"[Source {idx}: {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the MirAI Student Policy Advisor. Answer only from the retrieved handbook context. "
            "Do not use pretrained knowledge, assumptions, internet facts, or invented policies. "
            "If the context does not provide enough evidence, say the handbook context does not provide enough information. "
            "Never invent fines, email addresses, deadlines, exemptions, or disciplinary rules. "
            "Cite page numbers and section/source labels whenever available.",
        ),
        ("human", "Question:\n{question}\n\nRetrieved handbook context:\n{context}\n\nGrounded answer:"),
    ]
)


def build_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAG_PROMPT | get_llm() | StrOutputParser()
    return _rag_chain


def ingest_pdf_document(file_path: str, original_filename: str) -> Dict[str, Any]:
    global _vectorstore, _rag_chain
    _api_key()

    if not Path(file_path).exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not original_filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF uploads are supported.")

    try:
        documents = PyPDFLoader(file_path).load()
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc

    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError("The PDF has no extractable text.")

    for doc in documents:
        doc.metadata["source"] = original_filename
        if isinstance(doc.metadata.get("page"), int):
            doc.metadata["page_number"] = doc.metadata["page"] + 1

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    if not chunks:
        raise ValueError("The PDF produced zero retrievable chunks.")

    try:
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIRECTORY,
        )
        existing = store.get()
        if existing.get("ids"):
            store.delete(ids=existing["ids"])
        store.add_documents(chunks)
        stored_count = store._collection.count()
        if stored_count <= 0:
            raise RuntimeError("ChromaDB accepted ingestion but the collection is still empty.")
    except Exception as exc:
        raise RuntimeError(_friendly_google_error(exc, EMBEDDING_MODEL, "Embedding/Chroma ingestion")) from exc

    _vectorstore = store
    _rag_chain = None
    return {
        "success": True,
        "filename": original_filename,
        "page_count": len(documents),
        "chunk_count": len(chunks),
        "stored_document_count": stored_count,
        "message": f"Ingested {len(documents)} pages into {len(chunks)} chunks. ChromaDB now contains {stored_count} stored chunks.",
        "persist_directory": PERSIST_DIRECTORY,
    }


def answer_question(question: str) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not vectorstore_ready():
        raise RuntimeError("Vector store is empty. Upload or ingest the handbook PDF first.")

    try:
        store = get_vectorstore()
        llm = get_llm()
        base_retriever = store.as_retriever(search_kwargs={"k": 6})
        try:
            retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
            docs = retriever.invoke(question)
        except Exception:
            docs = base_retriever.invoke(question)
        unique_docs = []
        seen = set()
        for doc in docs:
            key = (doc.page_content, doc.metadata.get("page_number"))
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        context = format_docs(unique_docs)
        if not context.strip():
            return {
                "answer": "The handbook context does not provide enough information to answer this question.",
                "sources": [],
                "retrieved_context": "",
            }
        answer = build_rag_chain().invoke({"question": question, "context": context})
    except Exception as exc:
        raise RuntimeError(_friendly_google_error(exc, CHAT_MODEL, "RAG chat")) from exc

    sources = []
    for doc in unique_docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "MirAI handbook"),
                "page": doc.metadata.get("page_number"),
            }
        )
    return {"answer": answer, "sources": sources, "retrieved_context": context}


@app.get("/")
@app.get("/health")
def health():
    index = get_index_stats()
    return {
        "status": "online",
        "api": "ready",
        "vector_store_ready": index["ready"],
        "indexed_chunk_count": index["chunk_count"],
        "persist_directory": PERSIST_DIRECTORY,
        "default_handbook_found": DEFAULT_HANDBOOK_PATH.exists(),
        "default_handbook_name": DEFAULT_HANDBOOK_PATH.name if DEFAULT_HANDBOOK_PATH.exists() else None,
        "google_api_key": "configured" if os.getenv("GOOGLE_API_KEY") else "missing",
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: Optional[UploadFile] = File(None)):
    if file is None:
        if not DEFAULT_HANDBOOK_PATH.exists():
            raise HTTPException(status_code=404, detail=f"Default handbook not found: {DEFAULT_HANDBOOK_PATH.name}")
        try:
            return IngestResponse(
                **await run_in_threadpool(
                    ingest_pdf_document,
                    str(DEFAULT_HANDBOOK_PATH),
                    DEFAULT_HANDBOOK_PATH.name,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename = file.filename or "uploaded_handbook.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    temp_dir = tempfile.mkdtemp()
    temp_file = Path(temp_dir) / filename
    try:
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return IngestResponse(**await run_in_threadpool(ingest_pdf_document, str(temp_file), filename))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/ingest/default", response_model=IngestResponse)
async def ingest_default_handbook():
    if not DEFAULT_HANDBOOK_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Default handbook not found: {DEFAULT_HANDBOOK_PATH.name}")
    try:
        return IngestResponse(
            **await run_in_threadpool(
                ingest_pdf_document,
                str(DEFAULT_HANDBOOK_PATH),
                DEFAULT_HANDBOOK_PATH.name,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return ChatResponse(**await run_in_threadpool(answer_question, request.question))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/query", response_model=ChatResponse)
async def legacy_query(request: ChatRequest):
    return await chat(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
