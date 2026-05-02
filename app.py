import asyncio
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_STREAMLIT_DIR = Path(__file__).resolve().parent / "streamlit"
sys.path.insert(0, str(PROJECT_STREAMLIT_DIR))
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from src.document_loader import PolicyDocumentLoader
from src.rag_chain import HRPolicyRAG
from src.vector_store import PolicyVectorStore


BASE_DIR = PROJECT_STREAMLIT_DIR
DATA_DIR = BASE_DIR / "data"
POLICY_FILE = DATA_DIR / "policies.json"
HRIS_FILE = DATA_DIR / "hris.json"

st.set_page_config(page_title="HR Policy RAG Assistant", page_icon="🏢")


def ensure_dummy_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not POLICY_FILE.exists():
        POLICY_FILE.write_text(
            json.dumps(
                [
                    {
                        "id": "1",
                        "title": "Remote Work Policy",
                        "category": "Workplace",
                        "description": "Employees can work remotely 2 days a week.",
                        "status": "active",
                        "effective_year": 2024,
                    }
                ]
            ),
            encoding="utf-8",
        )

    if not HRIS_FILE.exists():
        HRIS_FILE.write_text(
            json.dumps(
                [
                    {
                        "employee_id": "E100",
                        "name": "John Doe",
                        "pto_balance": 12,
                        "pto_details": [],
                    }
                ]
            ),
            encoding="utf-8",
        )


def _source_fingerprint() -> str:
    policy_bytes = POLICY_FILE.read_bytes() if POLICY_FILE.exists() else b""
    hris_bytes = HRIS_FILE.read_bytes() if HRIS_FILE.exists() else b""
    return sha256(policy_bytes + b"::" + hris_bytes).hexdigest()


@st.cache_resource
def initialize_rag_system():
    ensure_dummy_data()
    loader = PolicyDocumentLoader(str(POLICY_FILE), str(HRIS_FILE))
    docs = loader.create_documents()
    hris_data = loader.load_hris_data()
    vector_store = PolicyVectorStore(index_dir=str(BASE_DIR / "index"))
    vector_store.build_or_load_index(docs, source_fingerprint=_source_fingerprint())
    return HRPolicyRAG(vector_store, hris_data)


if "rag_chain" not in st.session_state:
    if "OPENROUTER_API_KEY" not in os.environ:
        st.error("Please set OPENROUTER_API_KEY in your environment.")
        st.stop()
    st.session_state.rag_chain = initialize_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your HR AI assistant. Ask me about policies or your PTO.",
        }
    ]


with st.sidebar:
    st.header("Employee Settings")
    employee_id = st.text_input("Employee ID (optional)", placeholder="E100")
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your HR AI assistant. Ask me about policies or your PTO.",
            }
        ]
        st.rerun()

st.title("HR Policy Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about leave, PTO, and HR policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                answer = asyncio.run(
                    st.session_state.rag_chain.query(
                        question=prompt,
                        employee_id=employee_id if employee_id else None,
                    )
                )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                message = f"An error occurred: {exc}"
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
