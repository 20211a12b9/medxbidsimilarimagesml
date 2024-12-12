import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables to store model and index
model = None
preprocess = None
index = None
valid_image_links_df = None

@app.route('/', methods=['GET'])
def home():
    return "MedXBid Image Search API is running! Use /find-similar endpoint."

@app.route('/find-similar', methods=['POST'])
def find_similar():
    # Delay heavy imports to reduce initial memory footprint
    import torch
    import clip
    import faiss
    from PIL import Image

    # Check if the image is uploaded via file
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    # Load model only when needed
    global model, preprocess, index, valid_image_links_df
    if model is None:
        try:
            # Use CPU to reduce memory
            device = "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            # Load FAISS index
            index = faiss.read_index("image_index.faiss")
            
            # Load image links
            valid_image_links_df = pd.read_csv("valid_image_links.csv")
        except Exception as e:
            return jsonify({"error": f"Model initialization failed: {str(e)}"}), 500

    file = request.files['image']

    try:
        # Process image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        query_input = preprocess(img).unsqueeze(0).to(model.device)

        # Generate embedding
        with torch.no_grad():
            query_features = model.encode_image(query_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)

        # Search index
        distances, indices = index.search(query_features.cpu().numpy(), 3)

        # Get similar image URLs
        similar_images = [
            valid_image_links_df['image_url'].iloc[idx] 
            for idx in indices[0]
        ]

        # Clear memory
        del query_input, query_features
        torch.cuda.empty_cache()

        return jsonify({"similar_images": similar_images}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
