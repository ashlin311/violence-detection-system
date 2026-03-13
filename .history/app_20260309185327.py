from flask import Flask
app = Flask(__name__)

@app.router('/health')
def health():
    return {"status": "ok"}   

if __name__ == '__main__':
    app.run(debug=True, host='