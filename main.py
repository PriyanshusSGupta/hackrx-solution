import os
import io
import requests
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from pypdf import PdfReader

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough # New import
from langchain.schema.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain # New import
from langchain.chains import create_retrieval_chain # New import

# --- Environment and API Key Setup ---
load_dotenv()
HACKRX_TOKEN = os.getenv("HACKRX_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer based on the document")

class StructuredAnswer(BaseModel):
    question: str
    answer: str
    context: List[Document]

class AnswerResponse(BaseModel):
    answers: List[str]

# --- Document Processing Logic ---
def process_pdf_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text_content = "".join(page.extract_text() + " " for page in reader.pages if page.extract_text())
        if not text_content.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")
        return text_content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading document: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

# --- RAG Processor Class (LCEL Refactor) ---
class RAGProcessor:
    def __init__(self, document_text: str):
        self.vector_store = self._create_vector_store(document_text)
        self.retrieval_chain = self._create_retrieval_chain()

    def _create_vector_store(self, document_text: str):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(document_text)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        return FAISS.from_texts(chunks, embeddings_model)

    def _create_retrieval_chain(self):
        # Using the modern LCEL (LangChain Expression Language) approach
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        
        prompt_template = """
        You are an expert assistant for answering questions about policy documents.
        Based *only* on the context provided below, provide a concise and accurate answer to the user's question.
        If the context does not contain the information needed to answer, state clearly: "The provided document does not contain this information."

        <context>
        {context}
        </context>

        Question: {input}
        
        Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        # This chain combines the LLM and the prompt.
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        # This chain combines the retriever with the QA chain.
        # It automatically handles retrieving documents and passing them to the next chain.
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        return create_retrieval_chain(retriever, question_answer_chain)

    def get_answer(self, question: str) -> StructuredAnswer:
        result = self.retrieval_chain.invoke({"input": question})
        return StructuredAnswer(
            question=question,
            answer=result.get("answer", "Error generating answer."),
            context=result.get("context", [])
        )

# --- FastAPI App ---
app = FastAPI(title="HackRx 6.0 RAG API", description="An intelligent query-retrieval system.")

async def verify_token(req: Request):
    token = req.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing or invalid token format")
    provided_token = token.split("Bearer ")[1]
    if provided_token != HACKRX_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token")
    return True

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def run_submission(
    payload: QueryRequest,
    is_authenticated: bool = Depends(verify_token)
):
    print(f"Received request for document: {payload.documents}")
    try:
        document_text = process_pdf_from_url(payload.documents)
        rag_processor = RAGProcessor(document_text)
        final_answers = []
        for question in payload.questions:
            print(f"--- Processing question: '{question}' ---")
            structured_result = rag_processor.get_answer(question)
            print("\n[EXPLAINABILITY TRACE]")
            print(f"Question: {structured_result.question}")
            print(f"Answer: {structured_result.answer}")
            print("Source Context:")
            for i, doc in enumerate(structured_result.context):
                print(f"  [{i+1}] {doc.page_content[:150]}...")
            print("------------------------\n")
            final_answers.append(structured_result.answer)
        return AnswerResponse(answers=final_answers)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/")
def read_root():
    return {"status": "API is running"}