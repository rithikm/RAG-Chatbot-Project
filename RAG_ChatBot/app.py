from flask import Flask, render_template, request, jsonify
from ragchatbot.rag_chain import RAG_chain
import os 
from config.config import GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

app = Flask(__name__)

qa_chain = RAG_chain()

@app.route("/")
def default():
 return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
 user_input = request.json.get("question")
 if not user_input:
    return jsonify({"error": "No question provided."}), 400
 response = qa_chain({"query": user_input})
 answer = response["result"]
 sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]

 return jsonify({"answer": answer, "sources": sources})

if __name__ == "__main__":
    app.run(debug=True)
    
