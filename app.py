from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO("best.pt")

def preprocess(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    return img

@app.route("/")
def home():
    return "Drone AI Server Running Successfully 🚀"

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    img = preprocess(img)

    results = model(img)[0]

    if len(results.boxes) == 0:
        return jsonify({"drone": "Unknown", "confidence": 0})

    cls = int(results.boxes.cls[0])
    conf = float(results.boxes.conf[0])

    drone_names = {
        0: "Autel EVO II",
        1: "DJI Mavic 3",
        2: "Parrot Anafi"
    }

    return jsonify({
        "drone": drone_names.get(cls, "Unknown"),
        "confidence": round(conf * 100, 2)
    })

if __name__ == "__main__":
    app.run()
