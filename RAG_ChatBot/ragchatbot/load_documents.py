from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(folder_path):
  loader = PyPDFDirectoryLoader(folder_path)
  documents = loader.load()
  if not documents:
    print(f"No documents found in {folder_path}. Please ensure there are PDF files in this folder.")
    return []
  
  for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ").strip()
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  return splitter.split_documents(documents)