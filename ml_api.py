from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

model = YOLO("best.pt")  # your trained model

drone_specs = {
    "phantom4": {
        "Range": "7 km",
        "Speed": "72 km/h",
        "Payload": "1.3 kg",
        "Battery": "30 min"
    },
    "anafi": {
        "Range": "4 km",
        "Speed": "55 km/h",
        "Payload": "500 g",
        "Battery": "25 min"
    }
}

@app.route("/predict", methods=["POST"])
def predict():

    if len(request.files) == 0:
        return jsonify({"error": "No file received"}), 400

    file = list(request.files.values())[0]
    filepath = "temp.jpg"
    file.save(filepath)

    results = model(filepath)
    predicted_class = results[0].names[results[0].probs.top1]
    confidence = float(results[0].probs.top1conf)

    specs = drone_specs.get(predicted_class, {})

    return jsonify({
        "model": predicted_class,
        "confidence": round(confidence * 100, 2),
        "range": specs.get("Range", "-"),
        "speed": specs.get("Speed", "-"),
        "payload": specs.get("Payload", "-"),
        "battery": specs.get("Battery", "-")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)