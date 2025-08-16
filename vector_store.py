# vector_store.py
import pdfplumber, docx, os
import openai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()
# La API key se configura automáticamente al inicializar el cliente
client = openai.OpenAI() 

INDEX_PATH = "chatbot.index"
TEXTS_PATH = "texts.pkl"
texts = []
metadatas = []

if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
    print(f"Cargando índice desde {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    print(f"✅ Índice con {index.ntotal} vectores y textos cargados correctamente.")
else:
    print("⚠️  Advertencia: No se encontraron archivos 'chatbot.index' y 'texts.pkl'.")
    print("El chatbot funcionará sin contexto. Ejecuta 'build_index.py' para crearlos.")
    index = faiss.IndexFlatL2(1536)

def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text], 
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def process_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def process_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def add_document_to_index(path):
    if path.endswith(".pdf"):
        content = process_pdf(path)
    elif path.endswith(".docx"):
        content = process_docx(path)
    else:
        return
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
    for chunk in chunks:
        if chunk.strip():
            emb = get_embedding(chunk)
            index.add(np.array([emb]).astype("float32"))
            texts.append(chunk)
            metadatas.append({"source": path})

def search_similar_chunks(query):
    if index.ntotal == 0:
        return ["No hay documentos cargados en la base de datos de vectores."]
    
    emb = get_embedding(query).astype("float32")
    D, I = index.search(np.array([emb]), k=3)
    return [texts[i] for i in I[0]]
