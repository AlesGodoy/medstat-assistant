# app.py
from flask import Flask, request
from chatbot import generate_response
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Diccionario para guardar el historial de conversación por número de teléfono
# En una aplicación real, esto podría guardarse en una base de datos más persistente.
chat_histories = {}

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.form.get("Body", "Hola")
    from_number = request.form.get("From")
    print(f"Mensaje de {from_number}: {incoming_msg}")

    # Obtiene el historial para este usuario, o crea uno nuevo si es la primera vez.
    user_history = chat_histories.get(from_number, [])

    # Llama a la función de respuesta, pasándole la pregunta Y el historial.
    response_text, updated_history = generate_response(incoming_msg, user_history)

    # Guarda el historial actualizado para el próximo mensaje.
    chat_histories[from_number] = updated_history

    xml_response = f"<Response><Message>{response_text}</Message></Response>"
    return xml_response, 200, {"Content-Type": "application/xml"}

if __name__ == "__main__":
    app.run(port=5000, debug=True)
