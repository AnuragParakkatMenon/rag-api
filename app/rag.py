import os
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI
from app.vector_store import VectorStore

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Vector store (384 = embedding dimension)
vector_store = VectorStore(dim=384)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    vector_store.add(embeddings, chunks)

def query_rag(question: str) -> str:
    query_embedding = embedding_model.encode(question)
    contexts = vector_store.search(query_embedding)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below in the language provided by the user by default usee English.
If the answer is not in the context, say "I don't know".

Context:
{contexts}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
