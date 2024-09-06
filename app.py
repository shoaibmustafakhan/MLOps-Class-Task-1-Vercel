from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Preprocessing function
def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((144, 144))  # Resize to match input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define prediction route (POST only)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_b64 = data.get('image')
        
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(image)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Handle favicon requests (to prevent 404)
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for favicon

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
