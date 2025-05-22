import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# LangChain and model setup
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize model and prompt chain
model = OllamaLLM(model="llama3.2")

#Prompt
#You are an assistant helping a retail company, TestWarehouse, analyze its business operations. TestWarehouse operates multiple stores globally and sells a variety of products ranging from electronics to clothing. It tracks detailed information about customers, employees, sales, products, suppliers, inventory, deliveries, store issues, and ratings.


# Prompt template for TestWarehouse business analysis
template = """
You are an assistant helping a retail company, TestWarehouse, analyze its operations.
TestWarehouse runs global stores selling products from electronics to clothing.
It tracks data on customers, employees, sales, suppliers, inventory, deliveries, issues, and ratings.

Relevant reviews: {reviews}
Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ---------- UI Setup ----------
class TerminalWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Q&A Data")
        self.root.geometry("800x500")
        self.root.configure(bg='black')

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
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})

        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, f"\n🎤 You: {question}\n🤖 Answer: {result}\n\n> ")
        self.text_area.see(tk.END)

# ---------- Run App ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = TerminalWindow(root)
    root.mainloop()
