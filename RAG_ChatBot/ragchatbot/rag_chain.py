from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate  # use langchain.prompts, not langchain_core
from langchain_google_genai import GoogleGenerativeAI
from ragchatbot.vector_store import vector_store

def RAG_chain():
    llm = GoogleGenerativeAI(model="gemini-2.5-flash")

    # Load vector store
    vectorstores = vector_store()
    if vectorstores is None:  # fixed the check from 'vector_store' to 'vectorstores'
        print("Vector store is None. Cannot create the RAG chain.")
        return None

    # Create retriever
    retriever = vectorstores.as_retriever(search_kwargs={"k": 3})

    # Define prompt template
    prompt = PromptTemplate.from_template(
        "Answer the question based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Create the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",             # type of chain to use internally
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
