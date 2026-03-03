from ultralytics import YOLO
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model = YOLO("best.pt")

@app.route("/")
def home():
    return "Drone AI Server Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = "temp.jpg"
    file.save(filepath)

    results = model(filepath)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return jsonify({"drone": "No Drone Detected", "confidence": 0})

    class_id = int(boxes.cls[0])
    confidence = float(boxes.conf[0])
    drone_name = results[0].names[class_id]

    return jsonify({
        "drone": drone_name,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)