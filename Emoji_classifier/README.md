# Emoji Recognition Flask App

## Description
This is a Flask-based web application that classifies emojis using a Convolutional Neural Network (CNN) model built with TensorFlow/Keras. The app allows users to upload an image of an emoji, processes it, and predicts the corresponding emoji label.

## Features
- Upload an emoji image for recognition.
- Uses a trained CNN model (`emoji_recognition_cnn.h5`) to predict the emoji.
- Returns the corresponding emoji label in JSON format.
- Built with Flask for easy web deployment.

## Prerequisites
Ensure you have Python installed (recommended version: Python 3.8+).

## Installation & Setup
Follow these steps to set up and run the project:

### 1. Clone the Repository
```sh
git clone https://github.com/kailai-13/emoji-recognition-flask.git
cd emoji-recognition-flask
```

### 2. Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Place the Model File
Ensure that the `emoji_recognition_cnn.h5` model file is placed in the project directory.

### 5. Run the Flask Application
```sh
python app.py
```

### 6. Access the Web Interface
Once the server is running, open your browser and go to:
```
http://127.0.0.1:5000/
```

### 7. API Endpoint Usage
#### **POST /predict**
- Accepts a JSON payload with a base64-encoded image.
- Returns the predicted emoji.

Example Request (Using Python):
```python
import requests
import base64

# Read and encode image
with open("emoji.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

data = {"image": f"data:image/png;base64,{encoded_string}"}
response = requests.post("http://127.0.0.1:5000/predict", json=data)
print(response.json())
```

## Dependencies
See `requirements.txt` for the list of dependencies.

## Notes
- Ensure the model file (`emoji_recognition_cnn.h5`) is correctly trained and matches the labels provided in the script.
- Modify `emoji_names` in `app.py` if necessary to match the model's class labels.

## License
This project is open-source and can be modified and distributed freely.

