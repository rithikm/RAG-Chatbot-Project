from langchain_huggingface import HuggingFaceEmbeddings
import os
#Created a emebddings function which handles the embedding section of the RAG pipeline
#*NOTE*: GenerativeAIEmbedding which is Geminis Embedding model has a quota limit for the free teir which has been utilized so resorting to using the Hugging Face Embedding
def embeddings():
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  print("Using Hugging Face Embedding on CPU...")
  return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
