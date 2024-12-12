import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from PIL import Image
import io

try:
    import torch
    import clip
except ModuleNotFoundError:
    raise ImportError("torch or clip module is not installed. Please ensure these packages are installed in your environment.")

try:
    import faiss
except ModuleNotFoundError:
    raise ImportError("faiss module is not installed. Please ensure faiss is installed in your environment.")

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Load the CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
except Exception as e:
    raise RuntimeError(f"Failed to load CLIP model: {str(e)}")

# Load the FAISS index
try:
    index = faiss.read_index("image_index.faiss")
except FileNotFoundError:
    raise FileNotFoundError("image_index.faiss not found. Please ensure it is present in the working directory.")
except Exception as e:
    raise RuntimeError(f"Failed to load FAISS index: {str(e)}")

# Load valid image links
try:
    valid_image_links_df = pd.read_csv("valid_image_links.csv")
    if 'image_url' not in valid_image_links_df.columns:
        raise ValueError("CSV file does not contain 'image_url' column.")
except FileNotFoundError:
    raise FileNotFoundError("valid_image_links.csv not found. Please ensure it is present in the working directory.")
except Exception as e:
    raise RuntimeError(f"Error loading valid_image_links.csv: {str(e)}")

# Define the find-similar endpoint
@app.route('/find-similar', methods=['POST'])
def find_similar():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']

    try:
        # Open the uploaded image
        img = Image.open(io.BytesIO(file.read()))
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

if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0")
