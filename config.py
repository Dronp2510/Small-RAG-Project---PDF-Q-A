
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
