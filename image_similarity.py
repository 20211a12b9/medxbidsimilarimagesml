import os
import faiss
import torch
import clip
import pandas as pd
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import io

# Configuration
EXCEL_FILE = 'mexbidimagesdata.xlsx'
INDEX_FILE = 'image_index.faiss'
LINKS_FILE = 'valid_image_links.csv'
DOWNLOAD_DIR = 'downloaded_images'

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def prepare_image_embeddings():
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Read the Excel file
    data = pd.read_excel(EXCEL_FILE)

    # Specify the column containing image links
    image_links = data['IMAGE LINK']

    # Initialize lists for storing embeddings and valid links
    embeddings = []
    valid_image_links = []

    # Download and preprocess images
    for i, link in enumerate(image_links):
        try:
            # Download the image
            response = requests.get(link, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")  # Ensure RGB format

            # Preprocess the image for CLIP
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Generate image embedding
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize
                embeddings.append(image_features.cpu().numpy())

            # Save valid image link
            valid_image_links.append(link)

            print(f"Processed image {i+1}/{len(image_links)}")

        except Exception as e:
            print(f"Error processing image {i+1}: {e}")

    # Ensure embeddings are not empty
    if not embeddings:
        raise ValueError("No valid images were processed. Please check the input links.")

    # Convert embeddings to a NumPy array
    embeddings = np.vstack(embeddings)

    # Build a FAISS index
    dimension = embeddings.shape[1]  # Feature dimension (512 for CLIP ViT-B/32)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the FAISS index and links for later use
    faiss.write_index(index, INDEX_FILE)
    pd.DataFrame({'image_url': valid_image_links}).to_csv(LINKS_FILE, index=False)

    print("Indexing complete.")

def create_flask_app():
    # Load the CLIP model and preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load the FAISS index
    index = faiss.read_index(INDEX_FILE)

    # Load valid image links
    valid_image_links_df = pd.read_csv(LINKS_FILE)
    
    app = Flask(__name__)
    CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

    @app.route('/find-similar', methods=['POST'])
    def find_similar():
        # Check if the image is uploaded via file
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

    return app

def main():
    # IMPORTANT: Uncomment this ONLY ONCE to create initial index
    # prepare_image_embeddings()
    
    app = create_flask_app()
    app.run(port=5000, host="0.0.0.0")

if __name__ == "__main__":
    main()