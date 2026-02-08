import os
import shutil
from fastapi import FastAPI, UploadFile, File
from app.models import QueryRequest, QueryResponse
from app.rag import ingest_pdf, query_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG API")

UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from fastapi import FastAPI

app = FastAPI()



# Add this CORS middleware
origins = [
    "http://localhost:5173",  # your dev frontend
    "https://rag-api-production-4ec9.up.railway.app/"  # production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # allow your frontend domain
    allow_credentials=True,
    allow_methods=["*"],         # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"]          # allow Content-Type, Authorization, etc.
)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest-pdf")
async def ingest_pdf_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(file_path)
    return {"message": "PDF ingested successfully"}

@app.post("/query", response_model=QueryResponse)
def query_api(request: QueryRequest):
    answer = query_rag(request.question)
    return {"answer": answer}
