import os
import io
import uuid
import time
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

if not CHROMA_API_KEY:
    raise ValueError("CHROMA_API_KEY is missing")

if not CHROMA_TENANT:
    raise ValueError("CHROMA_TENANT is missing")

if not CHROMA_DATABASE:
    raise ValueError("CHROMA_DATABASE is missing")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Research Paper Assistant"
)


# =========================================================
# CHROMA CLOUD
# =========================================================

chroma_client = chromadb.CloudClient(
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
    api_key=CHROMA_API_KEY
)

collection = chroma_client.get_or_create_collection(
    name="research_papers"
)


# =========================================================
# TEXT SPLITTER
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


# =========================================================
# QUESTION MODEL
# =========================================================

class Question(BaseModel):
    question: str


# =========================================================
# GEMINI EMBEDDING
# =========================================================

def create_embedding(text: str, task_type: str):

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-embedding-001:embedContent"
    )

    payload = {
        "model": "models/gemini-embedding-001",
        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        },
        "taskType": task_type,
        "outputDimensionality": 768
    }

    for attempt in range(3):

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        # Success
        if response.status_code == 200:

            data = response.json()

            return data["embedding"]["values"]

        # Temporary server problem
        if response.status_code in [429, 500, 502, 503, 504]:

            if attempt < 2:
                time.sleep(3)
                continue

        raise Exception(
            f"Gemini Embedding Error "
            f"{response.status_code}: {response.text}"
        )

    raise Exception(
        "Gemini embedding service is temporarily unavailable."
    )


# =========================================================
# GEMINI ANSWER GENERATION
# =========================================================

def generate_answer(prompt: str):

    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-3.6-flash:generateContent"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # Retry up to 3 times for temporary errors
    for attempt in range(3):

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        # -----------------------------------------
        # Success
        # -----------------------------------------

        if response.status_code == 200:

            data = response.json()

            try:

                return (
                    data["candidates"][0]
                    ["content"]["parts"][0]["text"]
                )

            except (KeyError, IndexError):

                raise Exception(
                    "Gemini returned an unexpected response."
                )

        # -----------------------------------------
        # Temporary errors
        # -----------------------------------------

        if response.status_code in [429, 500, 502, 503, 504]:

            if attempt < 2:

                time.sleep(3)
                continue

        # -----------------------------------------
        # Other errors
        # -----------------------------------------

        raise Exception(
            f"Gemini API Error "
            f"{response.status_code}: {response.text}"
        )

    raise Exception(
        "Gemini is temporarily unavailable. "
        "Please try again in a few seconds."
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
def home():

    return FileResponse("index.html")


# =========================================================
# UPLOAD PDF DOCUMENTS
# =========================================================

@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...)
):

    uploaded = []
    errors = []

    for file in files:

        # -----------------------------------------
        # Check file type
        # -----------------------------------------

        if not file.filename.lower().endswith(".pdf"):

            errors.append(
                f"{file.filename}: "
                "Only PDF files are supported."
            )

            continue

        try:

            # -----------------------------------------
            # Read PDF
            # -----------------------------------------

            file_bytes = await file.read()

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            documents = []

            # -----------------------------------------
            # Extract text page by page
            # -----------------------------------------

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

            # -----------------------------------------
            # Check readable content
            # -----------------------------------------

            if not documents:

                errors.append(
                    f"{file.filename}: "
                    "No readable text found."
                )

                continue

            # -----------------------------------------
            # Split into chunks
            # -----------------------------------------

            chunks = splitter.split_documents(
                documents
            )

            ids = []
            texts = []
            embeddings = []
            metadatas = []

            # -----------------------------------------
            # Create embedding for each chunk
            # -----------------------------------------

            for chunk in chunks:

                embedding = create_embedding(
                    chunk.page_content,
                    "RETRIEVAL_DOCUMENT"
                )

                ids.append(
                    str(uuid.uuid4())
                )

                texts.append(
                    chunk.page_content
                )

                embeddings.append(
                    embedding
                )

                metadatas.append(
                    {
                        "source": chunk.metadata.get(
                            "source",
                            file.filename
                        ),
                        "page": chunk.metadata.get(
                            "page",
                            1
                        )
                    }
                )

            # -----------------------------------------
            # Save to Chroma Cloud
            # -----------------------------------------

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
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

    # =====================================================
    # RESPONSE
    # =====================================================

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


# =========================================================
# CHAT / RAG
# =========================================================

@app.post("/chat")
def chat(data: Question):

    try:

        # -----------------------------------------
        # Validate question
        # -----------------------------------------

        question = data.question.strip()

        if not question:

            return {
                "answer": "Please enter a question.",
                "sources": []
            }

        # -----------------------------------------
        # Convert question to embedding
        # -----------------------------------------

        question_embedding = create_embedding(
            question,
            "RETRIEVAL_QUERY"
        )

        # -----------------------------------------
        # Search Chroma Cloud
        # -----------------------------------------

        results = collection.query(
            query_embeddings=[
                question_embedding
            ],
            n_results=5,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        # -----------------------------------------
        # No documents found
        # -----------------------------------------

        if not documents:

            return {
                "answer": (
                    "The answer is not available "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        # -----------------------------------------
        # Build retrieved context
        # -----------------------------------------

        context_parts = []

        for text, metadata in zip(
            documents,
            metadatas
        ):

            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
DOCUMENT: {source}
PAGE: {page}

{text}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # -----------------------------------------
        # RAG prompt
        # -----------------------------------------

        prompt = f"""
You are a Research Paper Assistant.

Your job is to answer the user's question
using ONLY the provided research paper context.

STRICT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT make assumptions.
5. Do NOT invent information.
6. If the answer is not present in the context,
respond exactly:

The answer is not available in the uploaded documents.

7. Keep the answer clear and concise.
8. Do not create fake citations.
9. Use the source information provided in the context.

================================
RESEARCH PAPER CONTEXT
================================

{context}

================================
USER QUESTION
================================

{question}
"""

        # -----------------------------------------
        # Generate final answer
        # -----------------------------------------

        answer = generate_answer(
            prompt
        )

        # -----------------------------------------
        # Prepare sources
        # -----------------------------------------

        sources = []
        seen = set()

        for metadata in metadatas:

            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get(
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

        # -----------------------------------------
        # Final response
        # -----------------------------------------

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