from langchain_community.vectorstores import FAISS
import os 
from ragchatbot.embeddings import embeddings
from ragchatbot.load_documents import load_documents
from config.config import DATA_FOLDER, INDEX_FOLDER 



#Implementing the Vectorstore using FAISS Vectorstore Database
def vector_store():
  # Call the embeddings function and assign its return value to embeddings
  embeddings_model = embeddings()
  if os.path.exists(os.path.join(INDEX_FOLDER, "FAISS_Index")):
    print("Loading existing FAISS index...")
    # Add allow_dangerous_deserialization=True for loading
    return FAISS.load_local(INDEX_FOLDER, embeddings_model, allow_dangerous_deserialization=True)

  print(" Creating new FAISS index (first run)")
  # The documents are already split into chunks by load_documents, so no need to call split_documents again
  chunks = load_documents(DATA_FOLDER)
  if not chunks:
    print("No document chunks created. Cannot build the vectorstore")
    return None # Return None or raise an error if no chunks
  vectorstore = FAISS.from_documents(chunks, embeddings_model)
  vectorstore.save_local(INDEX_FOLDER)
  print("FAISS index saved successfully!")
  return vectorstore