# chatbot.py
import openai
from vector_store import search_similar_chunks

# Inicializamos el cliente aquí también
client = openai.OpenAI()

def generate_response(user_query):
    context_chunks = search_similar_chunks(user_query)
    context = "\n---\n".join(context_chunks)
    
    prompt = f"""Eres un asistente virtual llamado Medstat Assistant. Tu única tarea es responder a la pregunta del usuario basándote estrictamente en el siguiente contexto. Si la respuesta no se encuentra en el contexto, di "Lo siento, no tengo información sobre eso en mis documentos." No inventes información.

Contexto proporcionado:
{context}

Pregunta del usuario: {user_query}

Respuesta:"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error al llamar a la API de OpenAI: {e}")
        return "Hubo un problema al procesar tu solicitud. Por favor, intenta de nuevo."
