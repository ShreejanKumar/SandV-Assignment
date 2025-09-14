# Healthcare Document Chat

A Streamlit web app that lets you **chat with multiple healthcare documents** (PDF, DOCX, TXT).  
It uses **document embeddings**, **FAISS vector database**, and **LLM-based question answering** to retrieve relevant information from uploaded files.

## Features
- Upload one or more healthcare documents (PDF, DOCX, TXT)  
- Extract text including DOCX **tables and paragraphs**  
- Convert text into embeddings using **Google Embeddings API**  
- Store embeddings in **FAISS** for efficient search  
- Ask questions and get answers **based on uploaded documents**  
- Display **relevant excerpts** with **unique sources**  
- Handles **follow-up questions** with context maintained  

## Setup & Installation
1. **Clone the repo**
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>

2. **Install dependencies**
pip install -r requirements.txt

3. **Set API keys** in `.streamlit/secrets.toml`
[secrets]
LLAMAPARSE_API_KEY = "YOUR_LLAMAPARSE_KEY"
GOOGLE_API_KEY = "YOUR_GOOGLE_KEY"

4. **Run the app**
streamlit run app.py

## Usage
1. Upload healthcare documents.  
2. Wait for text extraction and embeddings.  
3. Ask questions in chat input and get answers from documents.  
4. View **relevant excerpts** in the expandable section for each answer.

## Project Structure
- app.py # Main Streamlit app
- helper.py # Helper functions
- extracted/ # Extracted text files (not committed)
- requirements.txt # Python dependencies
- .gitignore # Ignore env files, etc.
- README.md

## Notes
- Use anonymized/test documents only; no real patient data.  
- Includes medical disclaimer: **for demo/educational purposes only**, not a replacement for professional medical advice.
