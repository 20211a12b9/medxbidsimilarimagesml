import os
import io
import gc
import torch
import faiss
import clip
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Global configuration
MAX_SIMILAR_IMAGES = 3
FAISS_INDEX_PATH = "image_index.faiss"
VALID_LINKS_CSV_PATH = "valid_image_links.csv"

class ImageSimilaritySearcher:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.index = None
        self.valid_image_links_df = None

    def load_resources(self):
        """Load model, index, and image links with memory efficiency"""
        # Load model only when needed
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load FAISS index
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        
        # Use low-memory CSV reading
        self.valid_image_links_df = pd.read_csv(
            VALID_LINKS_CSV_PATH, 
            low_memory=True, 
            usecols=['image_url']
        )

    def encode_image(self, image):
        """Generate memory-efficient image embedding"""
        # Preprocess and move to device
        query_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding with reduced precision
        with torch.no_grad():
            query_features = self.model.encode_image(query_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            
            # Reduce precision to float16
            query_features = query_features.half().cpu().numpy()
        
        return query_features

    def find_similar_images(self, image):
        """Find similar images with memory management"""
        try:
            # Encode image
            query_features = self.encode_image(image)
            
            # Search FAISS index
            distances, indices = self.index.search(query_features, MAX_SIMILAR_IMAGES)
            
            # Retrieve image URLs
            similar_images = [
                self.valid_image_links_df['image_url'].iloc[idx] 
                for idx in indices[0]
            ]
            
            return similar_images
        
        finally:
            # Explicit memory clearing
            torch.cuda.empty_cache()
            gc.collect()

def create_app():
    """Factory function to create Flask app"""
    app = Flask(__name__)
    CORS(app)  # Enable Cross-Origin Resource Sharing
    
    # Initialize searcher
    searcher = ImageSimilaritySearcher()
    searcher.load_resources()

    @app.route('/find-similar', methods=['POST'])
    def find_similar():
        # Validate image upload
        if 'image' not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        
        try:
            # Read and process image
            file = request.files['image']
            img = Image.open(io.BytesIO(file.read()))
            
            # Find similar images
            similar_images = searcher.find_similar_images(img)
            
            return jsonify({"similar_images": similar_images}), 200
        
        except Exception as e:
            # Comprehensive error handling
            return jsonify({
                "error": f"Error processing image: {str(e)}",
                "details": str(e)
            }), 500

    return app

# Application entry point
app = create_app()

if __name__ == '__main__':
    app.run(
        port=int(os.environ.get('PORT', 5000)), 
        host="0.0.0.0", 
        debug=False
    )
