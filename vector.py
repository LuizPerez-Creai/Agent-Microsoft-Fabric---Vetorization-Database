import os
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import atexit
import shutil
import time

# Enable caching with SQLite for persistence
cache_path = ".langchain.db"
if os.path.exists(cache_path):
    try:
        os.remove(cache_path)  # Clear old cache on startup
    except Exception as e:
        print(f"Warning: Could not remove cache file: {e}")
set_llm_cache(SQLiteCache(database_path=cache_path))

# === CONFIGURATION ===
PARQUET_FOLDER = "./parquet_data"
DB_LOCATION = "./chrome_langchain_db"
COLLECTION_NAME = "client_reviews"
EMBEDDING_MODEL = "mxbai-embed-large"
BATCH_SIZE = 100  # Process documents in batches

# === SETUP ===
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    num_ctx=4096,  # Increase context window
    num_thread=4   # Use multiple threads
)

vector_store = None

def get_vector_store():
    global vector_store
    # Ensure the directory is clean before creating a new store
    if os.path.exists(DB_LOCATION):
        try:
            # First try to close any existing connections
            if vector_store is not None:
                try:
                    vector_store._client.close()
                except:
                    pass
                vector_store = None
            
            # Wait a bit to ensure connections are closed
            time.sleep(1)
            
            # Now try to remove the directory
            shutil.rmtree(DB_LOCATION)
        except Exception as e:
            print(f"Warning: Could not remove existing DB directory: {e}")
            # If we can't remove it, try to continue anyway
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )
    return vector_store

# Register cleanup function
def cleanup():
    global vector_store
    try:
        if vector_store is not None:
            try:
                vector_store._client.close()
            except:
                pass
            vector_store = None
            print("✅ Vector store cleaned up")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

atexit.register(cleanup)

def get_table_schema(df):
    """Get a human-readable description of the table schema"""
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = str(df[col].iloc[0]) if not df[col].empty else "N/A"
        schema.append(f"Column: {col}, Type: {dtype}, Sample: {sample}")
    return "\n".join(schema)

def process_batch(documents, ids, start_idx):
    """Process a batch of documents"""
    end_idx = min(start_idx + BATCH_SIZE, len(documents))
    batch_docs = documents[start_idx:end_idx]
    batch_ids = ids[start_idx:end_idx]
    try:
        print(f"[VECTOR] Intentando agregar {len(batch_docs)} documentos al vector store...")
        vector_store.add_documents(batch_docs, ids=batch_ids)
        print(f"[VECTOR] Agregados {len(batch_docs)} documentos exitosamente.")
    except Exception as e:
        print(f"❌ Error al agregar documentos al vector store: {e}")
    return end_idx

def load_parquet_files(folder_path):
    documents = []
    ids = []
    
    # First, create a document with table schemas
    schema_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".parquet"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_parquet(file_path)
                schema = get_table_schema(df)
                schema_doc = Document(
                    page_content=f"Table: {filename}\nSchema:\n{schema}",
                    metadata={"type": "schema", "source": filename},
                    id=f"schema_{filename}"
                )
                print(f"[VECTOR] Agregando esquema: {schema_doc.metadata}")
                schema_docs.append(schema_doc)
            except Exception as e:
                print(f"❌ Failed to read schema for {filename}: {e}")
    
    # Add schema documents first
    documents.extend(schema_docs)
    
    # Then process the actual data
    for filename in os.listdir(folder_path):
        if filename.endswith(".parquet"):
            file_path = os.path.join(folder_path, filename)
            print(f"📥 Loading: {file_path}")

            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"❌ Failed to read {filename}: {e}")
                continue

            for i, row in df.iterrows():
                # Create a structured content with column names
                content_parts = []
                for col in df.columns:
                    if pd.notna(row[col]):
                        content_parts.append(f"{col}: {row[col]}")
                
                content = "\n".join(content_parts)
                doc_id = f"{filename}_{i}"

                # Create metadata from all columns
                metadata = {col: str(row[col]) for col in df.columns if pd.notna(row[col])}
                metadata["source"] = filename
                metadata["row_index"] = i

                doc = Document(
                    page_content=content,
                    metadata=metadata,
                    id=doc_id
                )
                print(f"[VECTOR] Agregando fila: {doc.metadata}")
                documents.append(doc)
                ids.append(doc_id)
    
    return documents, ids

def build_vector_db():
    global vector_store
    try:
        # First ensure cleanup
        cleanup()
        
        # Wait a bit to ensure all connections are closed
        time.sleep(1)
        
        if os.path.exists(DB_LOCATION):
            try:
                shutil.rmtree(DB_LOCATION)
            except Exception as e:
                print(f"Warning: Could not remove DB directory: {e}")
                # Try to continue anyway
        
        if os.path.exists('.langchain.db'):
            try:
                os.remove('.langchain.db')
            except Exception as e:
                print(f"Warning: Could not remove cache file: {e}")
        
        print("⚙️ Creando base vectorial...")
        vector_store = get_vector_store()
        docs, ids = load_parquet_files(PARQUET_FOLDER)
        total_docs = len(docs)
        print(f"[VECTOR] Total de documentos a vectorizar: {total_docs}")
        current_idx = 0
        while current_idx < total_docs:
            current_idx = process_batch(docs, ids, current_idx)
            print(f"✅ Procesados {current_idx}/{total_docs} documentos")
        print("✅ Vector DB creada desde cero.")
        
        # Verifica cuántos documentos hay en el vector store
        try:
            num_docs = len(vector_store.get()['ids'])
            print(f'🔎 Vector store contiene {num_docs} documentos después de la vectorización.')
        except Exception as e:
            print(f'❌ No se pudo contar los documentos después de la vectorización: {e}')
    except Exception as e:
        print(f"Error during vector DB build: {e}")
        raise

# Initialize vector store
vector_store = get_vector_store()

# Only add documents if DB is not initialized yet
if not os.path.exists(DB_LOCATION):
    build_vector_db()
else:
    print("✅ Vector DB already exists. Skipping rebuild.")

# Create retriever for search with optimized settings
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,  # Number of results to return
        "filter": None  # Can be used to filter results
    }
)

if __name__ == "__main__":
    build_vector_db()