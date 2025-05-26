import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import re
import os
import shutil
from vector import retriever, build_vector_db, cleanup, vector_store
import atexit

# LangChain and model setup
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Register cleanup function
atexit.register(cleanup)

# Initialize model and prompt chain
model = OllamaLLM(model="llama3.2")


template = """
You are an assistant helping a retail company, TestWarehouse, analyze its operations.
TestWarehouse runs global stores selling products from electronics to clothing.
It tracks data on customers, employees, sales, suppliers, inventory, deliveries, issues, and ratings.

If a table schema is provided below, ALWAYS use only the columns listed in the schema for your answer. Do not invent or assume columns.

Table Schema (if available):
{reviews}

Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Aumenta k para mejorar la recuperación de esquemas
def get_relevant_docs(question):
    # Recupera más documentos para asegurar que el esquema esté incluido
    return retriever.invoke(question, config={"search_kwargs": {"k": 15}})

def extract_schema_doc(docs, table_name):
    for doc in docs:
        if (
            doc.metadata.get('type') == 'schema' and
            doc.metadata.get('source') == f"{table_name}.parquet"
        ):
            return doc
    return None

def get_schema_doc_direct(table_name):
    # Busca directamente el documento de esquema en el vector store
    results = vector_store.get(where={"type": "schema", "source": f"{table_name}.parquet"})
    if results and results['documents']:
        return results['documents'][0]
    return None

def extract_table_names(question):
    # Busca todas las palabras después de "columna", "columnas", "tabla", "table"
    raw_tables = re.findall(r"(?:columna|columnas|tabla|table)\\s+([A-Za-z0-9_]+)", question, re.IGNORECASE)
    # Filtra palabras comunes que no son nombres de tabla
    stopwords = {"de", "la", "las", "el", "los", "un", "una", "del", "al"}
    return [t for t in raw_tables if t.lower() not in stopwords]

# ---------- UI Setup ----------
class TerminalWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Q&A Data")
        self.root.geometry("800x500")
        self.root.configure(bg='black')
        
        # Add cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.text_area = ScrolledText(
            root, wrap=tk.WORD, bg="black", fg="white", insertbackground="black", font=("Courier", 14)
        )
        self.text_area.pack(expand=True, fill="both")
        self.text_area.insert(tk.END, "🎬 Movie Q&A Assistant\nType your question and press Enter.\nType 'q' to quit.\n\n> ")
        self.text_area.bind("<Return>", self.process_input)
        self.text_area.focus()

        self.buffer = ""

    def process_input(self, event):
        content = self.text_area.get("1.0", tk.END)
        lines = content.strip().split("\n")
        last_line = lines[-1]

        # Extract input after the last prompt
        if '> ' in last_line:
            user_input = last_line.split("> ", 1)[1].strip()
        else:
            user_input = last_line.strip()

        if user_input.lower() == "q":
            self.root.destroy()
            return "break"

        # Disable editing during processing
        self.text_area.config(state=tk.DISABLED)
        self.root.after(100, self.handle_question, user_input)

        return "break"  # prevent newline

    def handle_question(self, question):
        reviews = get_relevant_docs(question)
        print("\n--- DEBUG: Documentos recuperados ---")
        for i, doc in enumerate(reviews):
            print(f"Doc {i}: id={getattr(doc, 'id', None)}, type={doc.metadata.get('type')}, source={doc.metadata.get('source')}, metadatos={doc.metadata}")
        print("--- FIN DEBUG ---\n")
        question_lower = question.lower()
        pregunta_es_sobre_columnas = any(word in question_lower for word in ["column", "columna", "campo", "field"])
        table_names = extract_table_names(question)
        # Construye el contexto con todos los esquemas relevantes
        schema_texts = []
        for table in table_names:
            schema_doc = extract_schema_doc(reviews, table)
            if not schema_doc:
                schema_doc = get_schema_doc_direct(table)
            if schema_doc:
                schema_texts.append(f'[{table}]\n{schema_doc.page_content}')
        context = ''
        if schema_texts:
            context += '\n\n'.join(schema_texts) + '\n\n'
        # Incluye los reviews normales
        context += "\n\n".join([doc.page_content for doc in reviews if doc.metadata.get('type') != 'schema'])

        # Si la pregunta es sobre columnas y hay solo una tabla, responde como antes
        if pregunta_es_sobre_columnas and len(table_names) == 1:
            schema_doc = None
            if schema_texts:
                schema_doc = get_schema_doc_direct(table_names[0])
            if schema_doc:
                schema_text = schema_doc.page_content
                columnas = re.findall(r"Column: ([^,]+)", schema_text)
                if columnas:
                    columnas_str = ", ".join(columnas)
                    result = f"Las columnas de la tabla \"{table_names[0]}\" son:\n\n" + '\n'.join([f"{i+1}. {col}" for i, col in enumerate(columnas)])
                else:
                    result = "No se pudieron extraer los nombres de columna del esquema."
            else:
                result = "No se encontró el esquema de la tabla para mostrar las columnas."
        else:
            # Llama al LLM con el contexto y la pregunta
            result = chain.invoke({"reviews": context, "question": question})

        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"\n🎤 You: {question}\n🤖 Answer: {result}\n\n> ")
        self.text_area.see(tk.END)

    def on_closing(self):
        cleanup()
        self.root.destroy()

# ---------- Run App ----------
if __name__ == "__main__":
    try:
        # Build vector DB if needed
        build_vector_db()
        
        # Verifica cuántos documentos hay en el vector store
        try:
            num_docs = len(vector_store.get()['ids'])
            print(f'🔎 Vector store contiene {num_docs} documentos.')
        except Exception as e:
            print(f'❌ No se pudo contar los documentos: {e}')
        
        # Create and run the main window
        root = tk.Tk()
        app = TerminalWindow(root)
        root.mainloop()
    except Exception as e:
        print(f"Error during startup: {e}")
        cleanup()
