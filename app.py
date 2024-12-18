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
        self.device = "cpu"  # Force CPU to reduce memory usage
        self.model = None
        self.preprocess = None
        self.index = None
        self.valid_image_links = None
        self.max_similar = max_similar

    def load_resources(self):
        """Load resources with minimal memory footprint"""
        # Load model in CPU mode
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # Use memory-mapped FAISS index
        self.index = faiss.read_index("image_index.faiss", faiss.IO_FLAG_MMAP)

        # Use minimal memory DataFrame loading
        self.valid_image_links = pd.read_csv(
            "valid_image_links.csv",
            usecols=['image_url'],
            dtype={'image_url': str},
            low_memory=True
        )['image_url'].tolist()

    def encode_image(self, image):
        """Generate extremely memory-efficient image embedding"""
        # Resize image to reduce memory consumption
        image = image.resize((112, 112))  # Smaller resolution

        # Preprocess with minimal memory overhead
        query_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use lowest precision possible
            query_features = self.model.encode_image(query_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)

            # Convert to most memory-efficient format
            query_features = query_features.float().cpu().numpy().astype(np.float16)

        return query_features

    def find_similar_images(self, image):
        """Find similar images with extreme memory conservation"""
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
            gc.collect()

# Global function to manage memory-efficient resource loading
def get_resources():
    if not hasattr(g, 'resources'):
        g.resources = MemoryEfficientImageSearcher()
        g.resources.load_resources()
    return g.resources

def create_app():
    app = Flask(__name__)
    # CORS(app)
    CORS(app, resources={
        r"/find-similar": {
            "origins": [
                "http://localhost:8081",  # Your local development frontend
                "https://medxbidsimilarimagesml.onrender.com"  # Your production frontend
            ],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    @app.route('/find-similar', methods=['POST'])
    def find_similar():
        resources = get_resources()
        print("starting---")
        # Check for uploaded image
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        print("middle-----")
        try:
            # Open and process the uploaded image
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            similar_images = resources.find_similar_images(img)

            return jsonify({
                "similar_images": similar_images,
                "count": len(similar_images)
            }), 200
            print("print(similar_images)",similar_images)
        
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    @app.route("/", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "running"}), 200

    return app

    @app.route("/", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "running"}), 200

    return app

# Application initialization
app = create_app()

if __name__ == '__main__':
    # Use gunicorn for better memory management
    # Run with: gunicorn -w 1 -b 0.0.0.0:5000 app:app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=False  # Disable threading to reduce memory
    )
