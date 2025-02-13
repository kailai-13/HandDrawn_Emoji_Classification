# Emoji Classifier using Google QuickDraw Dataset

## Overview
This project is an emoji classifier trained using the [Google QuickDraw dataset](https://quickdraw.withgoogle.com/data). The model can recognize hand-drawn emojis and classify them into different categories based on the dataset.

## Features
- Uses the Google QuickDraw dataset for training.
- Implements deep learning models such as CNNs for classification.
- Supports real-time emoji recognition from hand-drawn sketches.
- Trained using TensorFlow/Keras or PyTorch.
- Interactive UI using a web or mobile app for real-time predictions.

## Dataset
The dataset consists of hand-drawn sketches of emojis collected through Google's QuickDraw initiative. It contains thousands of samples for each emoji category in `.ndjson` format.

### Dataset Preprocessing
1. Convert `.ndjson` files into images or NumPy arrays.
2. Normalize and reshape data for input into the neural network.
3. Split data into training and validation sets.

## Model Architecture
- **Convolutional Neural Network (CNN)** with multiple layers:
  - Convolutional layers for feature extraction
  - Pooling layers for dimensionality reduction
  - Fully connected layers for classification
- Uses ReLU activation and Softmax for output classification.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/kailai-13/emoji-classifier.git
   cd emoji-classifier
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the QuickDraw dataset:
   ```sh
   python download_dataset.py
   ```
4. Train the model:
   ```sh
   python train.py
   ```
5. Run the classifier:
   ```sh
   python app.py
   ```

## Usage
- **Training**: Run `train.py` to train the model on the dataset.
- **Inference**: Use `predict.py` to classify a given hand-drawn emoji.
- **Web App**: Start a Flask or Streamlit app with `app.py` for a user-friendly interface.

## Example Predictions
Upload or draw an emoji, and the classifier will predict the category.

## Future Enhancements
- Improve model accuracy with more advanced architectures like ResNet.
- Deploy the model as a web or mobile application.
- Add support for real-time sketch recognition.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is open-source and available under the MIT License.

