import os

# Load environment variables (optional: use python-dotenv)
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_FOLDER = "data"
INDEX_FOLDER = "FAISS_INDEX"