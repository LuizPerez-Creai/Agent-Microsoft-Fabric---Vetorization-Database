import os
import time
import tkinter as tk

def main():
    print("🌟 Starting LocalAIAgentSQL workflow...")
    
    try:
        # Step 1: Run database connection
        print("\n🚀 Running database connection...")
        print("📝 Downloading data from Azure and saving to CSV files")
        import databaseconection
        time.sleep(1)  # Small delay between scripts
        
        # Step 2: Run vector database creation
        print("\n🚀 Creating vector database...")
        print("📝 Processing CSV files and creating embeddings")
        import vector
        time.sleep(1)  # Small delay between scripts
        
        # Step 3: Run main interface
        print("\n🚀 Starting Q&A interface...")
        print("📝 Launching the interactive terminal")
        
        # Import the necessary components from main.py
        from main import TerminalWindow
        
        # Create and run the main window
        root = tk.Tk()
        app = TerminalWindow(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Error: Could not import required module - {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
