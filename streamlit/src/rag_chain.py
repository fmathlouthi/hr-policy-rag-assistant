import re
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_CHAT_MODEL


class HRPolicyRAG:
    def __init__(self, vector_store, hris_data: List[Dict]):
        self.vector_store = vector_store
        self.hris_data = hris_data
        self.llm = ChatOpenAI(
            model=OPENROUTER_CHAT_MODEL,
            temperature=0,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful HR assistant. Answer the user's question using only the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        )
        return prompt | self.llm | StrOutputParser()

    async def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        docs = await self.vector_store.similarity_search(query, k=k)
        return [{"metadata": doc.metadata, "content": doc.page_content} for doc in docs]

    async def call_hris_api(self, employee_id: str) -> Dict:
        employee_id = str(employee_id)
        for employee in self.hris_data:
            if str(employee.get("employee_id")) == employee_id:
                return employee
        return {"error": f"Employee ID {employee_id} not found"}

    async def query(self, question: str, employee_id: Optional[str] = None) -> str:
        docs = await self.get_relevant_documents(question)
        question_lc = question.lower()
        detected_employee = re.search(r"\bE\d{3,}\b", question.upper())
        resolved_employee_id = employee_id or (detected_employee.group(0) if detected_employee else None)

        personal_keywords = ("my ", "mine", "me ", "myself", "i ", "pto", "balance", "vacation", "leave")
        is_personal_query = any(keyword in question_lc for keyword in personal_keywords)

        # Prevent mixing another employee's HRIS record into personal answers.
        if is_personal_query and resolved_employee_id:
            docs = [doc for doc in docs if doc["metadata"].get("type") != "hris"]

        context_parts = [doc["content"] for doc in docs]

        if resolved_employee_id and is_personal_query:
            hris_info = await self.call_hris_api(resolved_employee_id)
            if "error" not in hris_info:
                asks_leave_balance = (
                    ("day" in question_lc or "days" in question_lc)
                    and ("pto" in question_lc or "leave" in question_lc or "available" in question_lc)
                )
                if asks_leave_balance:
                    return (
                        f"You have {hris_info.get('pto_balance')} PTO days available "
                        f"(Employee ID: {hris_info.get('employee_id')})."
                    )
                context_parts.append(
                    "Personal Data for Requesting Employee:\n"
                    f"Employee ID: {hris_info.get('employee_id')}\n"
                    f"Name: {hris_info.get('name')}\n"
                    f"PTO Balance: {hris_info.get('pto_balance')}"
                )

        final_context = "\n\n---\n\n".join(context_parts)
        return await self.chain.ainvoke({"context": final_context, "question": question})
