# vector_store.py
import pdfplumber, docx, os
import openai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = openai.OpenAI()

PERSISTENT_DIR = "/data/index"
INDEX_PATH = os.path.join(PERSISTENT_DIR, "chatbot.index")
TEXTS_PATH = os.path.join(PERSISTENT_DIR, "texts.pkl")

os.makedirs(PERSISTENT_DIR, exist_ok=True)

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
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_density=1, y_density=1)
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error procesando el archivo PDF {file_path}: {e}")
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100000,
        chunk_overlap=2000,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)

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
    D, I = index.search(np.array([emb]), k=5) 
    return [texts[i] for i in I[0]]
