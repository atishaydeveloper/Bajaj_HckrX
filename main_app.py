import os
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Union
from tempfile import NamedTemporaryFile
import requests
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("API_KEY", "d229e5eb06d6a264c4cebecd4fb0dc33e6a81c7bfa1f01945f751424fcac1e3a")


def concise_answer(ans):
    return ans.replace('\n', ' ')[:200] + ("..." if len(ans) > 200 else "")


def process_rag(pdf_path: str, questions: List[str]):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectordb = FAISS.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    chat_history = []
    answers = []
    for q in questions:
        result = chain({"question": q, "chat_history": chat_history})
        answer = result["answer"]
        answers.append(concise_answer(answer))
        chat_history.append((q, answer))
    return answers



# === Pydantic Model for Request ===
class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# === Main HackRX Endpoint ===
@app.post("/hackrx/run")
async def run_hackrx(
    request: HackRxRequest,
    authorization: str = Header(None, alias="Authorization")
):
    """Main HackRX endpoint with required request/response format and authentication."""
    # --- Authorization check ---
    if not authorization:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing Authorization header")
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

    # Handle both single document and multiple documents
    document_urls = request.documents if isinstance(request.documents, list) else [request.documents]
    all_answers = []
    for doc_url in document_urls:
        # Download PDF
        try:
            r = requests.get(doc_url)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download file: {r.status_code}")
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")
        try:
            answers = process_rag(tmp_path, request.questions)
            all_answers.extend(answers)
        finally:
            os.remove(tmp_path)
    return {"answers": all_answers}
