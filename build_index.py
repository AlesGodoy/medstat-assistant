import os
import faiss
import pickle
from vector_store import add_document_to_index, texts, index

# Carpeta donde están los documentos
# Puedes cambiar este nombre si usas otra carpeta
# Por ahora, usaremos el directorio actual porque los PDFs están aquí

doc_folder = "Documentos"

documentos = [f for f in os.listdir(doc_folder) if f.endswith(".pdf") or f.endswith(".docx")]

if not documentos:
    print("No se encontraron documentos PDF o DOCX en la carpeta.")
else:
    for doc in documentos:
        print(f"Procesando: {doc}")
        add_document_to_index(os.path.join(doc_folder, doc))

    if texts:
        faiss.write_index(index, "chatbot.index")
        with open("texts.pkl", "wb") as f:
            pickle.dump(texts, f)
        print("\n✅ Índice FAISS y textos guardados correctamente.")
    else:
        print("No se añadieron textos al índice.")
