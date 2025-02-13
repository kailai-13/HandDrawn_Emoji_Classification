import tensorflow as tf
import numpy as np
import base64
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("emoji_recognition_cnn.h5")

# Emoji class names (must match training labels)
emoji_names = [
    "beaming-face -😁 ", "cloud - ☁️", "face-spiral -😵‍💫", "flushed-face - 😳",
    "grimacing-face - 😬", "grinning-face -😄", "grinning-squinting -😆",
    "heart - ❤️", "pouting-face -😡", "raised-eyebrow -🤨", "relieved-face -😌",
    "savoring-food -😋 ", "smiling-heart -😍", "smiling-horns",
    "smiling-sunglasses -😎", "smiling-tear -🥲", "smirking-face -😏", "tears-of-joy -😂"
]

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")

        # Convert to numpy array & preprocess
        image = image.resize((64, 64))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using model
        predictions = model.predict(image)[0]  # Get prediction array
        top_k = 3  # Number of top predictions to return
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_emojis = [emoji_names[i] for i in top_indices]
        top_probabilities = [predictions[i] * 100 for i in top_indices]  # Convert to percentage

        return jsonify({"emojis": top_emojis, "probabilities": top_probabilities})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)