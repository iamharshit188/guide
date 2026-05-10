import os
from flask import Flask, send_from_directory, jsonify, abort
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/docs/<path:filename>")
def serve_doc(filename):
    docs_dir = os.path.join(BASE, "docs")
    filepath = os.path.join(docs_dir, filename)
    if not os.path.realpath(filepath).startswith(os.path.realpath(docs_dir)):
        abort(403)
    if not os.path.isfile(filepath):
        abort(404)
    return send_from_directory(docs_dir, filename, mimetype="text/plain; charset=utf-8")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE, filename)


if __name__ == "__main__":
    print("Starting dev server at http://localhost:3000")
    app.run(debug=True, port=3000, host="0.0.0.0")
