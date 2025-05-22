import os
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# === CONFIGURATION ===
CSV_FOLDER = "./csv_data"
DB_LOCATION = "./chrome_langchain_db"
COLLECTION_NAME = "client_reviews"
EMBEDDING_MODEL = "mxbai-embed-large"

# === SETUP ===
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

def load_csv_files(folder_path):
    documents = []
    ids = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"📥 Loading: {file_path}")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")
                continue

            for i, row in df.iterrows():
                title = str(row.get("title", ""))
                review = str(row.get("review", ""))
                rating = str(row.get("raiting", row.get("rating", "")))
                date = str(row.get("date", ""))

                content = title + " " + review
                doc_id = f"{filename}_{i}"

                doc = Document(
                    page_content=content,
                    metadata={"rating": rating, "date": date, "source": filename},
                    id=doc_id
                )

                documents.append(doc)
                ids.append(doc_id)
    
    return documents, ids

# Only add documents if DB is not initialized yet
if not os.path.exists(DB_LOCATION):
    print("⚙️ Creating vector database...")
    docs, ids = load_csv_files(CSV_FOLDER)
    vector_store.add_documents(docs, ids=ids)
else:
    print("✅ Vector DB already exists. Skipping rebuild.")

# Create retriever for search
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
