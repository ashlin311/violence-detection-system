from flask import Flask
app = Flask(__name__)

@app.router('/health')
def health():
    return {"status": "ok"}   