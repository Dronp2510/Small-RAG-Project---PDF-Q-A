
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application configuration
APP_NAME = os.getenv("APP_NAME", "RAG PDF Chatbot")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# PDF configuration
PDF_PATH = os.getenv("PDF_PATH", "data/sample.pdf")

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Text chunking configuration (FR-03)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embedding configuration (FR-04)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "32")
)