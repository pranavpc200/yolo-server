from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the YOLOv8 model from the same folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'YOLOv8 Flask API is live ✅'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    print("📥 Received POST request at /predict")

    if 'image' not in request.files:
        print("❌ No image in request.files")
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = model(img)
        detections = results[0].names
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = [detections[c] for c in classes]

        print("🧠 Detected classes:", class_names)
        return jsonify({'predictions': class_names})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Only used for local testing. Render uses its own WSGI server.
    app.run(host='0.0.0.0', port=5000)
