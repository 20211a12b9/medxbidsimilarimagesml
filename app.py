import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Use a try-except block for imports
try:
    import torch
    import clip
    import faiss
    from PIL import Image
except ImportError as e:
    print(f"Import error: {e}")
    torch = clip = faiss = None

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Global variables to store model and index
model = None
preprocess = None
index = None
valid_image_links_df = None

def initialize_model():
    global model, preprocess, index, valid_image_links_df
    
    if torch is None or clip is None or faiss is None:
        raise ImportError("Required libraries are not imported correctly")
    
    # Load the CLIP model and preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Check if index files exist
    if not (os.path.exists("image_index.faiss") and os.path.exists("valid_image_links.csv")):
        raise FileNotFoundError("Index files are missing. Please run download_and_index.py first.")

    # Load the FAISS index
    index = faiss.read_index("image_index.faiss")

    # Load valid image links
    valid_image_links_df = pd.read_csv("valid_image_links.csv")

# Try to initialize on startup
try:
    initialize_model()
except Exception as e:
    print(f"Initialization error: {e}")

@app.route('/', methods=['GET'])
def home():
    return "MedXBid Image Search API is running!"

@app.route('/find-similar', methods=['POST'])
def find_similar():
    # Reinitialize model if not loaded
    global model, preprocess, index, valid_image_links_df
    if model is None:
        try:
            initialize_model()
        except Exception as e:
            return jsonify({"error": f"Model initialization failed: {str(e)}"}), 500

    # Check if the image is uploaded via file
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']

    try:
        # Open the uploaded image
        img = Image.open(io.BytesIO(file.read()))
        query_input = preprocess(img).unsqueeze(0).to(model.device)

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
