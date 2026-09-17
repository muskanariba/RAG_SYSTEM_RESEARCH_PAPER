# Research Paper Assistant using RAG

A simple Retrieval-Augmented Generation (RAG) application that allows users to upload research papers and ask questions based only on the uploaded documents.

## Technologies

- Python
- FastAPI
- LangChain
- ChromaDB
- OpenAI
- PyPDF
- HTML/CSS/JavaScript

## Features

- Upload one or multiple PDFs
- Extract PDF text
- Split text into chunks
- Generate embeddings
- Store embeddings in ChromaDB
- Retrieve relevant document chunks
- Ask questions about uploaded papers
- Display document name and page number
- Does not answer from outside knowledge
- Handles empty questions and invalid files

## Installation

```bash
pip install -r requirements.txt