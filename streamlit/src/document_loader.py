import json
import os
from hashlib import sha256
from datetime import datetime
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PolicyDocumentLoader:
    def __init__(self, policy_path: str, hris_path: str):
        self.policy_path = policy_path
        self.hris_path = hris_path

    def load_policies(self) -> List[Dict]:
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Policies file not found at {self.policy_path}")

        with open(self.policy_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("The loaded policies data is not a list")
        return data

    def load_hris_data(self) -> List[Dict]:
        if not os.path.exists(self.hris_path):
            raise FileNotFoundError(f"HRIS file not found at {self.hris_path}")

        with open(self.hris_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("The loaded HRIS data is not a list")
        return data

    def create_documents(self) -> List[Document]:
        documents: List[Document] = []
        current_year = datetime.now().year
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
        )

        for policy in self.load_policies():
            is_active = policy.get("status") == "active" or policy.get("is_active") is True
            is_current_year = policy.get("effective_year") == current_year
            if not (is_active or is_current_year):
                continue

            content = (
                f"Policy ID: {policy.get('id')}\n"
                f"Policy Title: {policy.get('title')}\n"
                f"Category: {policy.get('category')}\n"
                f"Description: {policy.get('description')}\n"
                f"Effective Year: {policy.get('effective_year')}"
            )
            policy_metadata = {
                "title": policy.get("title"),
                "id": policy.get("id"),
                "type": "policy",
                "category": policy.get("category"),
                "effective_year": policy.get("effective_year"),
            }
            chunks = splitter.create_documents([content], metadatas=[policy_metadata])
            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = idx
                chunk.metadata["chunk_count"] = len(chunks)
                chunk.metadata["doc_hash"] = sha256(
                    f"{policy.get('id')}::{idx}::{chunk.page_content}".encode("utf-8")
                ).hexdigest()
                documents.append(chunk)

        for employee in self.load_hris_data():
            content = (
                f"Employee ID: {employee.get('employee_id')}\n"
                f"Name: {employee.get('name')}\n"
                f"PTO Balance: {employee.get('pto_balance')}\n"
                f"PTO Details: {json.dumps(employee.get('pto_details', []))}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "title": f"HRIS Record: {employee.get('name')}",
                        "employee_id": employee.get("employee_id"),
                        "type": "hris",
                        "doc_hash": sha256(
                            f"{employee.get('employee_id')}::{content}".encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )

        return documents
