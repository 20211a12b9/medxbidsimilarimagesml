import os
import io
import gc
import torch
import faiss
import clip
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, g
from flask_cors import CORS

class MemoryEfficientImageSearcher:
    def __init__(self, max_similar=3):
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self.index = None
        self.valid_image_links = None
        self.max_similar = max_similar

    def load_resources(self):
        """Optimize resource loading"""
        # Use lower memory model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Use memory-mapped index with read-only mode
        self.index = faiss.read_index("image_index.faiss", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        
        # Minimal memory DataFrame loading
        self.valid_image_links = pd.read_csv(
            "valid_image_links.csv",
            usecols=['image_url'],
            dtype={'image_url': str},
            low_memory=True
        )['image_url'].tolist()

    def encode_image(self, image):
        """Ultra-low memory image encoding"""
        # Extremely small resize to minimize memory
        image = image.resize((64, 64))  # Even smaller resolution

        # Preprocess with minimal overhead
        query_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use lowest precision possible
            query_features = self.model.encode_image(query_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)

            # Most memory-efficient conversion
            query_features = query_features.float().cpu().numpy().astype(np.float16)

        return query_features

    def find_similar_images(self, image):
        """Minimal memory similar image search"""
        try:
            # Encode image with minimal memory
            query_features = self.encode_image(image)

            # Limit search to prevent excessive memory use
            distances, indices = self.index.search(query_features, self.max_similar)

            # Retrieve image URLs with minimal overhead
            similar_images = [
                self.valid_image_links[idx]
                for idx in indices[0]
                if 0 <= idx < len(self.valid_image_links)
            ]

            return similar_images

        except Exception as e:
            print(f"Search error: {e}")
            return []

        finally:
            # Aggressive memory cleanup
            del query_features
            torch.cuda.empty_cache()
            gc.collect()

def create_app():
    app = Flask(__name__)
    
    # Comprehensive CORS configuration
    CORS(app, resources={
        r"/find-similar": {
            "origins": ["*"],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })

    # Global cache for resources
    resources = MemoryEfficientImageSearcher()
    resources.load_resources()

    @app.route('/find-similar', methods=['POST'])
    def find_similar():
        # Check for uploaded image
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        try:
            # Open and process the uploaded image
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            similar_images = resources.find_similar_images(img)

            return jsonify({
                "similar_images": similar_images,
                "count": len(similar_images)
            }), 200

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    @app.route("/", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "running"}), 200

    return app

# Application initialization
app = create_app()

if __name__ == '__main__':
    # Use gunicorn with minimal workers
    # Run with: gunicorn -w 1 -b 0.0.0.0:5000 app:app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=False  # Disable threading to reduce memory
    )
