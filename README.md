
# Dermatology RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) assistant designed to answer dermatology-related questions using the *Dermatology Handbook, 3rd Edition* as its knowledge source. It was created as part of the Agentic AI Developer Certification (AAIDC) program.

## Overview

The assistant works by retrieving sections of the handbook that relate to a user's question and then generating an answer based strictly on those sections. This approach ensures that responses come from the handbook and not from outside assumptions.

## Key Features

- PDF Processing: Loads and processes the dermatology handbook.
- Text Chunking: Splits the text into manageable, overlapping sections.
- Vector Embeddings: Converts text into numerical vectors with Sentence-Transformers.
- Persistent Storage: Stores embeddings using ChromaDB.
- Semantic Search: Finds relevant text chunks based on the meaning of the user’s question.
- Context-Based Answering: Uses an LLM to generate answers based only on the retrieved context.
- Support for Multiple LLM Providers: Works with OpenAI, Groq, or Google Gemini depending on the available API key.

## Tech Stack

- Language: Python 3.9+
- Framework: LangChain
- Vector Database: ChromaDB
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- LLM Providers: OpenAI, Groq, Google Gemini
- Document Loader: PyPDFLoader
- Environment Management: python-dotenv

## System Architecture

### Ingestion Pipeline

1. Load the Dermatology Handbook PDF from the data/ directory.
2. Split the text into overlapping chunks.
3. Create embeddings for each chunk and store them in ChromaDB.
4. This step runs once and creates the persistent vector store.

### Query Pipeline

1. The user asks a question.
2. The question is converted into a vector using the same embedding model.
3. The system performs a similarity search to find the most relevant chunks.
4. These chunks and the original question are placed into a prompt template.
5. The LLM generates an answer based only on the retrieved material.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- An API key for OpenAI, Groq, or Google Gemini

### 1. Clone the Repository
```bash

git clone <https://github.com/10486-JosephMutua/dermatology-rag-assistant.git>
cd <dermatology-rag-assistant>
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables

Create a .env file:

```bash
Linux/macOS:
cp .env.example .env
```

```bash
Windows:
copy .env.example .env
```
Open .env and add one provider key:
```bash
# OPENAI_API_KEY="..."
# GROQ_API_KEY="..."
# GOOGLE_API_KEY="..."
```
### 4. Add the PDF Document

Place the handbook file into the data/ directory:

```txt
data/
└── Derm_Handbook_3rd-Edition-_Nov_2020-FINAL.pdf
```

## Running the Application

python src/app.py

On the first run, the system will process and embed the entire PDF. This may take a few minutes. Once done, the vector store is saved in chroma_db/, and later runs will start immediately.

## Example Questions

- "Describe the symptoms of psoriasis."
- "How is acne diagnosed?"
- "What is the difference between a papule and a pustule?"

## Project Structure


```txt
.
├── src/
│   ├── app.py           # Main application logic
│   └── vectordb.py      # ChromaDB interface
├── data/
│   └── Dermatology Handbook PDF
├── chroma_db/           # Persistent vector store
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
---

## License

This project is released under the MIT License.

## Acknowledgments

This project was developed following the guidelines and template provided by the ReadyTensor Agentic AI Developer Certification program.

