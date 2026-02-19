from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import mysql.connector
print("RAW PREDICTION:", preds)
print("ARGMAX:", np.argmax(preds))
print("CLASS:", class_names[np.argmax(preds)])

app = Flask(__name__)
CORS(app)

# -------------------------------
# DATABASE CONNECTION FUNCTION
# -------------------------------

def get_drone_info(drone_name):
    try:
        conn = mysql.connector.connect(
            host="localhost",        # <-- yahan apna host daalo
            user="u683932415_oetnab",        # <-- yahan apna username
            password="1Mpre$$10n@2023",    # <-- yahan apna password
            database="u683932415_oetnab"     # <-- yahan apna database name
        )

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM drones WHERE drone_name=%s", (drone_name,))
        result = cursor.fetchone()

        conn.close()
        return result

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# ROUTES
# -------------------------------

@app.route("/")
def home():
    return "Drone AI Server Running"


@app.route("/predict", methods=["POST"])
def predict():

    # Yahan future me AI image processing hoga
    # Abhi demo ke liye fixed drone name use kar rahe hain
    detected_drone = "DJI Mavic 3"

    drone_info = get_drone_info(detected_drone)

    if drone_info:
        return jsonify(drone_info)
    else:
        return jsonify({"error": "Drone not found"})


# -------------------------------
# RENDER PORT FIX
# -------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
from flask import Flask, jsonify
from flask_cors import CORS
import os

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
