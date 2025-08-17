# chatbot.py
import openai
import json
from vector_store import search_similar_chunks

client = openai.OpenAI()

# --- PLANTILLAS DE PROMPT PARA LA RESPUESTA FINAL ---
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


def analyze_user_intent(user_query, history):
    """
    Usa la IA para analizar la pregunta del usuario en su totalidad.
    - Corrige errores gramaticales y de tipeo.
    - Usa el historial para crear una pregunta autocontenida.
    - Determina si el usuario quiere una respuesta concreta o detallada.
    - Devuelve un objeto JSON estructurado con el análisis.
    """
    formatted_history = "\n".join([f"Usuario: {h['user']}\nAsistente: {h['bot']}" for h in history])

    analysis_prompt = f"""
    Analiza la 'Última pregunta del usuario' en el contexto de un 'Historial de conversación'. Realiza las siguientes tareas y devuelve el resultado EXCLUSIVAMENTE en formato JSON:

    1.  **Corrección:** Corrige cualquier error de tipeo, gramática o falta de acentos en la pregunta del usuario.
    2.  **Pregunta Autocontenida:** Reescribe la pregunta corregida para que sea una pregunta completa y que se entienda por sí sola, usando el historial para dar contexto si es necesario (por ejemplo, si la pregunta es 'y sus beneficios?', conviértela en '¿Cuáles son los beneficios de [tema anterior]?').
    3.  **Nivel de Detalle:** Determina si el usuario quiere una respuesta 'concreta' o 'detallada'. Por defecto es 'concreta', a menos que el usuario pida explícitamente más información, ampliar, o dar más detalles.

    **Historial:**
    {formatted_history}

    **Última pregunta del usuario:** "{user_query}"

    **Salida JSON (solo el JSON, sin texto adicional):**
    {{
      "standalone_query": "...",
      "detail_level": "..."
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"} # Forzamos la salida en JSON
        )
        analysis = json.loads(completion.choices[0].message.content)
        return analysis
    except Exception as e:
        print(f"Error al analizar la intención del usuario: {e}")
        # Si el análisis falla, devolvemos un resultado por defecto.
        return {
            "standalone_query": user_query,
            "detail_level": "concreta"
        }

def generate_response(user_query, history):
    # 1. Analizar la intención del usuario de forma inteligente
    intent_analysis = analyze_user_intent(user_query, history)
    standalone_query = intent_analysis.get("standalone_query", user_query)
    detail_level = intent_analysis.get("detail_level", "concreta")
    
    print(f"Pregunta original: '{user_query}' -> Pregunta analizada: '{standalone_query}' (Nivel: {detail_level})")

    # 2. Buscar en los documentos con la pregunta ya limpia y completa
    context_chunks = search_similar_chunks(standalone_query)
    context = "\n---\n".join(context_chunks)

    # 3. Seleccionar el prompt final basado en el análisis de intención
    if detail_level == "detallada":
        final_prompt = PROMPT_DETALLADO.format(context=context, user_query=user_query)
    else:
        final_prompt = PROMPT_CONCRETO.format(context=context, user_query=user_query)

    # 4. Generar la respuesta
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0
        )
        response_text = completion.choices[0].message.content

        # 5. Actualizar y devolver el historial
        updated_history = history[-3:] + [{"user": user_query, "bot": response_text}]
        return response_text, updated_history
        
    except Exception as e:
        print(f"Error al generar la respuesta final: {e}")
        return "Hubo un problema al procesar tu solicitud. Por favor, intenta de nuevo.", history
