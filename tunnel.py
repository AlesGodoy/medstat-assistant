from pyngrok import ngrok

# Abre el túnel en el puerto 5000
public_url = ngrok.connect(5000, "http")
print("🌐 URL pública de ngrok:")
print(public_url)
print("⬆️ Usá esa URL + /webhook en Twilio")

input("⏳ Presioná Enter para cerrar el túnel...")
ngrok.disconnect(public_url)
