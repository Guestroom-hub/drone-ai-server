from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Drone AI Server Running"

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({
        "drone": "DJI Mavic 3",
        "confidence": 0.95
    })

if __name__ == "__main__":
    app.run()
