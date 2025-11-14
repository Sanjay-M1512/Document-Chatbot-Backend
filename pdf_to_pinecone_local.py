import os
import time
import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitter import RecursiveCharacterTextSplitter

# ========= Settings =========
DEFAULT_MAX_CHARS = 1000   # chunk size (smaller = better semantic coherence)
DEFAULT_OVERLAP = 300     # overlap between chunks
BATCH_SIZE = 64
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim local embeddings
# ============================

def init_env():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "pdf-chunks-index")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing. Put it in .env")
    pc = Pinecone(api_key=api_key)
    return pc, index_name

def ensure_index(pc: Pinecone, index_name: str, dimension: int = 384, metric: str = "cosine"):
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while True:
            if pc.describe_index(index_name).status["ready"]:
                break
            time.sleep(2)
    return pc.Index(index_name)

def download_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def extract_pages(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages

def build_chunks_from_pages(pages, max_chars: int, overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    out = []
    for p in pages:
        sub_chunks = splitter.split_text(p["text"])
        for c in sub_chunks:
            out.append({"page": p["page"], "text": c})
    return out

def upsert_chunks(index, model, source_id: str, chunks: list, batch_size: int = BATCH_SIZE):
    print(f"Embedding {len(chunks)} chunks with {MODEL_NAME} and upserting to Pinecone...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        vecs = model.encode(texts, convert_to_numpy=True).tolist()
        vectors = []
        for j, (c, v) in enumerate(zip(batch, vecs)):
            vectors.append({
                "id": f"{source_id}::p{c['page']}::{i+j}",
                "values": v,
                "metadata": {
                    "text": c["text"],
                    "page": c["page"],
                    "source": source_id,
                    "chunk_id": i + j
                }
            })
        index.upsert(vectors=vectors)
    print("✅ Done.")

def process_pdf_to_pinecone(pdf_path=None, pdf_url=None):
    load_dotenv()
    max_chars = int(os.getenv("CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS))
    overlap = int(os.getenv("CHUNK_OVERLAP", DEFAULT_OVERLAP))

    # Init Pinecone + index
    pc, index_name = init_env()
    index = ensure_index(pc, index_name=index_name, dimension=384, metric="cosine")

    # Embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Load PDF bytes
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        source_id = f"local://{os.path.basename(pdf_path)}"
    elif pdf_url:
        pdf_bytes = download_pdf(pdf_url)
        source_id = pdf_url
    else:
        raise ValueError("No valid PDF path or URL provided.")

    # Extract + chunk
    pages = extract_pages(pdf_bytes)
    chunks = build_chunks_from_pages(pages, max_chars=max_chars, overlap=overlap)

    # Upsert
    upsert_chunks(index, model, source_id, chunks)

    print(f"✅ File ID for search: {source_id}")
    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "file_id": source_id
    }

def main():
    load_dotenv()
    pdf_path = os.getenv("PDF_PATH")
    pdf_url = os.getenv("PDF_URL")

    if not pdf_path and not pdf_url:
        pdf_path = input("Enter local PDF file path (or leave blank to use a URL): ").strip()
        if not pdf_path:
            pdf_url = input("Enter PDF URL: ").strip()

    max_chars = int(os.getenv("CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS))
    overlap = int(os.getenv("CHUNK_OVERLAP", DEFAULT_OVERLAP))

    pc, index_name = init_env()
    index = ensure_index(pc, index_name=index_name, dimension=384, metric="cosine")

    model = SentenceTransformer(MODEL_NAME)

    if pdf_path and os.path.exists(pdf_path):
        print(f"Reading local PDF: {pdf_path}")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        source_id = f"local://{os.path.basename(pdf_path)}"
    elif pdf_url:
        print(f"Downloading PDF from: {pdf_url}")
        pdf_bytes = download_pdf(pdf_url)
        source_id = pdf_url
    else:
        raise ValueError("No valid PDF path or URL provided.")

    print("Extracting pages...")
    pages = extract_pages(pdf_bytes)
    print(f"Extracted {len(pages)} pages with text.")

    print(f"Chunking pages (max_chars={max_chars}, overlap={overlap})...")
    chunks = build_chunks_from_pages(pages, max_chars=max_chars, overlap=overlap)
    print(f"Created {len(chunks)} chunks.")

    print("Upserting to Pinecone...")
    upsert_chunks(index, model, source_id, chunks)

if __name__ == "__main__":
    os.environ["USE_TF"] = "0"
    main()
