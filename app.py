import os
import faiss
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Force CPU device to prevent memory issues
device = "cpu"

try:
    # Explicitly specify device during model loading
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the FAISS index
    index = faiss.read_index("image_index.faiss")

    # Load valid image links
    valid_image_links_df = pd.read_csv("valid_image_links.csv")
    if 'image_url' not in valid_image_links_df.columns:
        raise ValueError("CSV file does not contain 'image_url' column.")

except Exception as e:
    print(f"Initialization error: {e}")
    model = preprocess = index = valid_image_links_df = None

@app.route('/find-similar', methods=['POST'])
def find_similar():
    # Check if all resources are loaded
    if model is None or preprocess is None or index is None or valid_image_links_df is None:
        return jsonify({"error": "Resources not initialized properly"}), 500

    # Check if the image is uploaded via file
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']

    try:
        # Open the uploaded image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        query_input = preprocess(img).unsqueeze(0).to(device)

        # Generate the embedding for the query image
        with torch.no_grad():
            query_features = model.encode_image(query_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)  # Normalize

        # Search the FAISS index
        k = 3  # Number of nearest neighbors
        distances, indices = index.search(query_features.cpu().numpy(), k)

        # Get the URLs of the most similar images
        similar_images = []
        for idx in indices[0]:
            image_path = valid_image_links_df['image_url'].iloc[idx]
            similar_images.append(image_path)

        return jsonify({"similar_images": similar_images}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    return "MedXBid Image Search API is running!"

if __name__ == '__main__':
    # Use a lower memory footprint
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)),
        threaded=False  # Disable threading to reduce memory usage
    )
