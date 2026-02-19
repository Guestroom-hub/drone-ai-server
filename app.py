from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

model = load_model("drone_model.h5")

class_names = ['DJI_Mavic_3', 'Autel_EVO_II', 'Parrot_Anafi']


@app.route("/")
def home():
    return "Drone AI Server Running"


@app.route("/predict", methods=["POST"])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']
    img_path = "temp.jpg"
    file.save(img_path)

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    drone_name = class_names[class_index]

    return jsonify({
        "drone": drone_name,
        "confidence": confidence
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
