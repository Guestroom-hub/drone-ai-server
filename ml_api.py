from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = YOLO("best.pt")  # make sure best.pt is in same folder

# =========================
# DRONE SPECIFICATIONS
# =========================
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
    },
    "mavic3": {
        "Range": "15 km",
        "Speed": "75 km/h",
        "Payload": "900 g",
        "Battery": "46 min"
    },
    "evo2": {
        "Range": "9 km",
        "Speed": "72 km/h",
        "Payload": "1 kg",
        "Battery": "40 min"
    }
}

# =========================
# ROOT ROUTE (IMPORTANT FOR RENDER)
# =========================
@app.route("/")
def home():
    return "Drone AI Server is Running 🚀"

# =========================
# HEALTH CHECK
# =========================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if len(request.files) == 0:
        return jsonify({"error": "No file received"}), 400

    file = list(request.files.values())[0]

    # unique temp filename
    temp_filename = f"{uuid.uuid4()}.jpg"
    file.save(temp_filename)

    try:
        results = model(temp_filename)

        # If classification model
        if results[0].probs is None:
            os.remove(temp_filename)
            return jsonify({"error": "Model is not classification type"}), 400

        predicted_class = results[0].names[results[0].probs.top1]
        confidence = float(results[0].probs.top1conf)

        # Confidence filter (below 50% reject)
        if confidence < 0.5:
            os.remove(temp_filename)
            return jsonify({
                "error": "Low confidence detection",
                "confidence": round(confidence * 100, 2)
            })

        specs = drone_specs.get(predicted_class.lower(), {})

        response = {
            "model": predicted_class,
            "confidence": round(confidence * 100, 2),
            "range": specs.get("Range", "-"),
            "speed": specs.get("Speed", "-"),
            "payload": specs.get("Payload", "-"),
            "battery": specs.get("Battery", "-")
        }

        os.remove(temp_filename)
        return jsonify(response)

    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
