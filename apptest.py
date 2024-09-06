import requests
import base64

# Read and encode an image
with open('F:/Nalaik_Semester/MLOps_A1/MLOps-Class-Task-1-Vercel/cats/cat.1.jpg', 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# Create JSON payload
payload = {'image': img_base64}

# Send POST request to the Flask API
response = requests.post('http://127.0.0.1:5000/predict', json=payload)

# Print the prediction response
print(response.json())
