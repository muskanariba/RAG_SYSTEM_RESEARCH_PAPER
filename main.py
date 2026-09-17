import os
import shutil
import requests
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# =========================
# ENV
# =========================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env"
    )


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="Research Paper Assistant"
)


# =========================
# FOLDERS
# =========================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# =========================
# LOCAL EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# CHROMA DB
# =========================

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# =========================
# TEXT SPLITTER
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


# =========================
# QUESTION MODEL
# =========================

class Question(BaseModel):
    question: str


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return FileResponse("index.html")


# =========================
# UPLOAD PDFS
# =========================

@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...)
):

    uploaded = []
    errors = []

    for file in files:

        # Only PDF
        if not file.filename.lower().endswith(".pdf"):

            errors.append(
                f"{file.filename}: Only PDF files are supported."
            )

            continue

        try:

            # Save file
            file_path = UPLOAD_DIR / file.filename

            with open(
                file_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            # Read PDF
            reader = PdfReader(
                str(file_path)
            )

            documents = []

            # Extract page text
            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""

                if text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file.filename,
                                "page": page_number
                            }
                        )
                    )

            # No readable text
            if not documents:

                errors.append(
                    f"{file.filename}: No readable text found."
                )

                continue

            # Split text
            chunks = splitter.split_documents(
                documents
            )

            # Store embeddings in Chroma
            vectorstore.add_documents(
                chunks
            )

            uploaded.append(
                file.filename
            )

        except Exception as e:

            print(
                "UPLOAD ERROR:",
                repr(e)
            )

            errors.append(
                f"{file.filename}: {str(e)}"
            )

    if not uploaded:

        return {
            "message": "Upload failed.",
            "files": [],
            "errors": errors
        }

    return {
        "message": (
            f"Successfully uploaded "
            f"{len(uploaded)} paper(s)."
        ),
        "files": uploaded,
        "errors": errors
    }


# =========================
# CHAT
# =========================

@app.post("/chat")
def chat(data: Question):

    try:

        question = data.question.strip()

        # Empty question
        if not question:

            return {
                "answer": "Please enter a question.",
                "sources": []
            }

        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

        # Nothing found
        if not docs:

            return {
                "answers":
                "The answer is not available in the uploaded documents.",
                "sources": []
            }

        # =========================
        # BUILD CONTEXT
        # =========================

        context_parts = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
DOCUMENT: {source}
PAGE: {page}

{doc.page_content}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # =========================
        # RAG PROMPT
        # =========================

        prompt = f"""
You are a Research Paper Assistant.

Answer the user's question ONLY using
the research paper context below.

STRICT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT make assumptions.
5. Do NOT invent information.
6. If the answer is not found in the
   context, respond exactly:

The answer is not available in the uploaded documents.

7. Keep the answer clear and concise.
8. Do not create fake citations.

========================
RESEARCH PAPER CONTEXT
========================

{context}

========================
USER QUESTION
========================

{question}
"""

        # =========================
        # GEMINI REST API
        # =========================

        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/gemini-3.6-flash:generateContent"
        )

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            timeout=60
        )

        # Gemini error
        if response.status_code != 200:

            print(
                "GEMINI ERROR:",
                response.text
            )

            return {
                "answer": (
                    f"Gemini API Error "
                    f"{response.status_code}: "
                    f"{response.text}"
                ),
                "sources": []
            }

        result = response.json()

        # Get answer
        answer = (
            result["candidates"][0]
            ["content"]["parts"][0]["text"]
        )

        # =========================
        # SOURCES
        # =========================

        sources = []
        seen = set()

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            key = (
                source,
                page
            )

            if key not in seen:

                seen.add(key)

                sources.append(
                    {
                        "document": source,
                        "page": page
                    }
                )

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        print(
            "CHAT ERROR:",
            repr(e)
        )

        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }