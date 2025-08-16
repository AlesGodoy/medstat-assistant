# chatbot.py
import openai
from vector_store import search_similar_chunks

client = openai.OpenAI()

# --- PLANTILLAS DE PROMPT ---
PROMPT_CONCRETO = """Eres un asistente virtual llamado Medstat Assistant. Tu tarea es responder a la pregunta del usuario de forma CONCRETA y DIRECTA, basándote estrictamente en el siguiente contexto. Si la respuesta no está en el contexto, di \"Lo siento, no tengo información sobre eso en mis documentos.\" No inventes información.

Contexto proporcionado:
{context}

Pregunta del usuario: {user_query}

Respuesta Concreta:"""

PROMPT_DETALLADO = """Eres un asistente virtual experto llamado Medstat Assistant. Tu tarea es responder a la pregunta del usuario de forma DETALLADA Y AMPLIA, utilizando toda la información relevante del siguiente contexto. Explica los puntos clave y proporciona tanto detalle como sea posible. Si la respuesta no está en el contexto, di \"Lo siento, no tengo información sobre eso en mis documentos.\" No inventes información.

Contexto proporcionado:
{context}

Pregunta del usuario: {user_query}

Respuesta Detallada:"""

def rephrase_question_with_history(query, history):
    """
    Usa el modelo de OpenAI para reformular la pregunta del usuario basándose
    en el historial, para que sea una pregunta completa y autocontenida.
    """
    if not history:
        return query # Si no hay historial, la pregunta ya es la primera.

    formatted_history = "\n".join([f"Usuario: {h['user']}\nAsistente: {h['bot']}" for h in history])
    
    prompt = f"""Dado el siguiente historial de conversación y la última pregunta del usuario, reescribe la última pregunta para que sea una pregunta autocontenida. Si la pregunta ya es autocontenida, devuélvela sin cambios.

Historial:
{formatted_history}

Última pregunta del usuario: \"{query}\"

Pregunta autocontenida:"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error al reformular la pregunta: {e}")
        return query

def generate_response(user_query, history):
    standalone_query = rephrase_question_with_history(user_query, history)
    print(f"Pregunta original: '{user_query}' -> Pregunta para búsqueda: '{standalone_query}'")

    context_chunks = search_similar_chunks(standalone_query)
    context = "\n---\n".join(context_chunks)

    palabras_clave_ampliar = ["más información", "amplía", "dame más detalles", "explica más"]
    
    if any(keyword in user_query.lower() for keyword in palabras_clave_ampliar):
        print("-> Usando prompt detallado.")
        final_prompt = PROMPT_DETALLADO.format(context=context, user_query=user_query)
    else:
        print("-> Usando prompt concreto.")
        final_prompt = PROMPT_CONCRETO.format(context=context, user_query=user_query)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0
        )
        response_text = completion.choices[0].message.content

        updated_history = history[-3:] + [{"user": user_query, "bot": response_text}]
        
        return response_text, updated_history
        
    except Exception as e:
        print(f"Error al llamar a la API de OpenAI: {e}")
        return "Hubo un problema al procesar tu solicitud. Por favor, intenta de nuevo.", history
