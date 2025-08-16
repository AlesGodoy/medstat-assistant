from flask import Flask, request
from chatbot import generate_response
from dotenv import load_dotenv

# Carga las variables de entorno (API keys, etc.)
load_dotenv()

# Simplemente inicializa la aplicación Flask
app = Flask(__name__)

# El archivo vector_store.py se encargará automáticamente
# de cargar el índice pre-construido cuando se importe.

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.form.get("Body", "Hola") # Usamos "Hola" por defecto si no hay mensaje
    from_number = request.form.get("From")
    print(f"Mensaje de {from_number}: {incoming_msg}")
    
    # Genera la respuesta usando el chatbot
    response = generate_response(incoming_msg)
    
    # Envía la respuesta en el formato XML que espera Twilio
    xml_response = f"<Response><Message>{response}</Message></Response>"
    return xml_response, 200, {"Content-Type": "application/xml"}

if __name__ == "__main__":
    # Esta parte es solo para pruebas locales
    app.run(port=5000, debug=True)
