"""Streamlit frontend for the MirAI Student Policy Advisor."""

from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

st.set_page_config(page_title="MirAI Student Policy Advisor", page_icon="M", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Plus Jakarta Sans', sans-serif;
  background: radial-gradient(circle at 10% 12%, rgba(0,240,255,.14), transparent 28%),
              radial-gradient(circle at 90% 18%, rgba(36,99,235,.18), transparent 30%),
              linear-gradient(135deg,#050816 0%,#07142d 52%,#030712 100%) !important;
  color: #e5f3ff;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: rgba(4,10,24,.86) !important; border-right: 1px solid rgba(34,211,238,.22); }
.mirai-wordmark { font-family:'Space Grotesk',sans-serif; font-size:2.25rem; font-weight:800; letter-spacing:4px;
  background:linear-gradient(120deg,#e0fbff,#22d3ee,#2563eb); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-align:center; margin: 0 0 2px; filter: drop-shadow(0 0 14px rgba(34,211,238,.35)); }
.brand-box,.panel { background:rgba(8,20,45,.72); border:1px solid rgba(34,211,238,.22); border-radius:16px; padding:20px;
  box-shadow:0 18px 45px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.04); animation: rise .45s ease both; }
@keyframes rise { from {opacity:0; transform:translateY(10px)} to {opacity:1; transform:translateY(0)} }
.hero { padding:30px 32px; border-radius:18px; background:linear-gradient(135deg,rgba(9,21,50,.92),rgba(5,13,31,.75));
  border:1px solid rgba(34,211,238,.28); box-shadow:0 0 42px rgba(34,211,238,.10); margin-bottom:20px; }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:2.7rem; margin:0; color:#f8fbff; letter-spacing:0; }
.hero p { color:#9bdcf2; font-size:1.05rem; margin:.5rem 0 0; }
.stButton>button, .stDownloadButton>button { width:100%; border-radius:10px !important; border:1px solid rgba(34,211,238,.34) !important;
  background:rgba(15,35,75,.72) !important; color:#eaf9ff !important; font-weight:700 !important; transition:.2s ease !important; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 0 22px rgba(34,211,238,.18); border-color:#22d3ee !important; }
.user-bubble,.ai-bubble { padding:14px 16px; border-radius:14px; margin:10px 0; border:1px solid rgba(34,211,238,.22); }
.user-bubble { margin-left:12%; background:linear-gradient(135deg,rgba(37,99,235,.36),rgba(14,165,233,.22)); }
.ai-bubble { margin-right:8%; background:rgba(7,18,41,.82); }
.source-chip { display:inline-block; margin:6px 6px 0 0; padding:4px 8px; border-radius:999px; background:rgba(34,211,238,.12); color:#8be9ff; font-size:.8rem; }
.small-muted { color:#8aa4bd; font-size:.86rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "nav" not in st.session_state:
    st.session_state.nav = "Home"
if "pending" not in st.session_state:
    st.session_state.pending = None
if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = None


def health():
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return res.ok, res.json() if res.ok else {"detail": res.text}
    except requests.RequestException as exc:
        return False, {"detail": str(exc)}


def api_error(resp):
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def ingest_default_handbook():
    try:
        res = requests.post(f"{BACKEND_URL}/ingest", timeout=180)
        if res.ok:
            data = res.json()
            st.session_state.ingest_message = ("success", f"{data['message']} Stored count: {data.get('stored_document_count', data['chunk_count'])}.")
        else:
            st.session_state.ingest_message = ("error", api_error(res))
    except requests.Timeout:
        st.session_state.ingest_message = ("error", "Ingestion timed out. The backend may still be processing; check /health in a moment.")
    except requests.RequestException as exc:
        st.session_state.ingest_message = ("error", f"Could not reach backend: {exc}")


def ask(question):
    st.session_state.messages.append({"role": "user", "content": question, "time": datetime.now().strftime("%H:%M")})
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json={"question": question}, timeout=TIMEOUT)
        if not resp.ok:
            raise RuntimeError(api_error(resp))
        data = resp.json()
        content = data.get("answer", "No answer returned by the backend.")
        sources = data.get("sources", [])
    except requests.Timeout:
        content, sources = "The backend timed out while searching the handbook. Try again after ingestion completes.", []
    except requests.RequestException:
        content, sources = "The FastAPI backend is unavailable. Start it with `python3 -m uvicorn backend:app --reload`.", []
    except Exception as exc:
        content, sources = f"The backend could not answer yet: {exc}", []
    st.session_state.messages.append({"role": "assistant", "content": content, "sources": sources, "time": datetime.now().strftime("%H:%M")})


with st.sidebar:
    st.markdown('<div class="brand-box"><div class="mirai-wordmark">MIRAI</div><div class="small-muted" style="text-align:center">School of Technology</div></div>', unsafe_allow_html=True)
    st.session_state.nav = st.radio("Navigation", ["Home", "Chat", "Policy Categories", "About"], index=["Home", "Chat", "Policy Categories", "About"].index(st.session_state.nav))
    ok, info = health()
    st.markdown("---")
    if ok:
        st.success("Backend online")
    else:
        st.warning("Backend unavailable")
    if ok:
        ready = info.get("vector_store_ready")
        chunks = info.get("indexed_chunk_count", 0)
        st.caption(f"Vector store ready: {ready} ({chunks} chunks)")
        st.caption(f"Model: {info.get('chat_model')}")
        if info.get("default_handbook_found") and st.button("Ingest bundled handbook"):
            ingest_default_handbook()
            st.rerun()
    if st.session_state.ingest_message:
        level, message = st.session_state.ingest_message
        if level == "success":
            st.success(message)
        else:
            st.error(message)
    uploaded = st.file_uploader("Upload handbook PDF", type=["pdf"])
    if uploaded and st.button("Ingest uploaded PDF"):
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            res = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=180)
            if res.ok:
                data = res.json()
                st.session_state.ingest_message = ("success", f"{data['message']} Stored count: {data.get('stored_document_count', data['chunk_count'])}.")
            else:
                st.session_state.ingest_message = ("error", api_error(res))
            st.rerun()
        except requests.RequestException as exc:
            st.session_state.ingest_message = ("error", f"Upload failed: {exc}")
            st.rerun()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown('<section class="hero"><h1>Your Policy Guide, Always</h1><p>Source-grounded answers from the MirAI School of Technology Student Policy Handbook 2026.</p></section>', unsafe_allow_html=True)

quick_questions = [
    "I have 72% attendance. How many attendance marks will I get?",
    "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?",
    "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?",
    "How much is the fine for smoking a cigarette on campus?",
]

if st.session_state.pending:
    q = st.session_state.pending
    st.session_state.pending = None
    ask(q)
    st.session_state.nav = "Chat"
    st.rerun()

if st.session_state.nav == "Home":
    ok, info = health()
    if ok and info.get("default_handbook_found") and not info.get("vector_store_ready"):
        st.markdown('<div class="panel"><h3>Bundled handbook found</h3><p class="small-muted">Click the sidebar button to ingest Mirai_SoT_Policy_Handbook_2026.pdf, then ask a question.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><h3>Ask the handbook</h3><p class="small-muted">Ingest the official PDF, then use chat for policy questions with page-backed references.</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}"):
                st.session_state.pending = question
                st.rerun()

elif st.session_state.nav == "Chat":
    st.subheader("Policy Chat")
    for msg in st.session_state.messages:
        klass = "user-bubble" if msg["role"] == "user" else "ai-bubble"
        label = "You" if msg["role"] == "user" else "MirAI Policy AI"
        st.markdown(f'<div class="{klass}"><div class="small-muted">{label} - {msg.get("time", "")}</div>{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            chips = "".join(f'<span class="source-chip">{s.get("source", "handbook")} p.{s.get("page", "?")}</span>' for s in msg["sources"])
            st.markdown(chips, unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Ask a question", placeholder="Example: What happens if my attendance is 72%?")
        if st.form_submit_button("Send") and question.strip():
            ask(question.strip())
            st.rerun()

elif st.session_state.nav == "Policy Categories":
    cols = st.columns(2)
    categories = ["Attendance & Evaluation", "Medical Leave", "Clubs & Societies", "Campus Conduct"]
    for i, item in enumerate(categories):
        with cols[i % 2]:
            st.markdown(f'<div class="panel"><h3>{item}</h3><p class="small-muted">Questions in this category are answered only from retrieved handbook sections.</p></div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="panel"><h3>About</h3><p>This course project uses FastAPI, Streamlit, LangChain LCEL, MultiQueryRetriever, Gemini, and persistent ChromaDB. It is intentionally grounded: when the PDF context is insufficient, the assistant should say so instead of guessing.</p></div>', unsafe_allow_html=True)
