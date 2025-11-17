from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import your functions
from pdf_to_pinecone_local import process_pdf_to_pinecone
from query_answer_gemini import query_pinecone_and_answer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---- 1. Upload PDF & Process into Chunks ----
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():

    # Thunder Client sends file as a list
    files = request.files.getlist("file")

    # Validate file input
    if not files or len(files) == 0:
        return jsonify({"error": "No file provided"}), 400

    file = files[0]   # take the first file uploaded

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Process PDF → Pinecone
    try:
        process_pdf_to_pinecone(filepath)
        return jsonify({"message": "PDF processed and stored in Pinecone", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- 2. Query Answer from Gemini + Pinecone ----
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        answer, retrieved_chunks = query_pinecone_and_answer(query)
        return jsonify({
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks  # for debugging
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
