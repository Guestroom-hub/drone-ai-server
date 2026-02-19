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

    # -------- LOAD MODEL --------
    from tensorflow.keras.models import load_model
    import numpy as np
    from tensorflow.keras.preprocessing import image

    model = load_model("model.h5")   # <-- tumhara trained model
    class_names = ["DJI Mavic 3", "Autel EVO II Pro", "Parrot Anafi"]

    # -------- IMAGE PROCESS --------
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    index = np.argmax(preds)
    confidence = float(preds[0][index])

    detected_drone = class_names[index]

    return jsonify({
        "drone": detected_drone,
        "confidence": confidence
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
