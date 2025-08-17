# app.py
import os
import json
from flask import Flask, request
from chatbot import generate_response
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

HISTORY_DIR = "/data/index/history"
os.makedirs(HISTORY_DIR, exist_ok=True)

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.form.get("Body", "Hola")
    from_number = request.form.get("From")
    print(f"Mensaje de {from_number}: {incoming_msg}")

    history_file_path = os.path.join(HISTORY_DIR, f"{from_number}.json")
    user_history = []

    if os.path.exists(history_file_path):
        try:
            with open(history_file_path, 'r', encoding='utf-8') as f:
                user_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            user_history = []
    
    response_text, updated_history = generate_response(incoming_msg, user_history)

    try:
        with open(history_file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error al guardar el historial para {from_number}: {e}")

    xml_response = f"<Response><Message>{response_text}</Message></Response>"
    return xml_response, 200, {"Content-Type": "application/xml"}

if __name__ == "__main__":
    app.run(port=5000, debug=True)
