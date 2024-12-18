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

class MemoryEfficientImageSearcher:
    def __init__(self, max_similar=3):
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self.index = None
        self.valid_image_links = None
        self.max_similar = max_similar

    def load_resources(self):
        """Ultra-lightweight resource loading"""
        # Use smaller model variant if possible
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Ensure index is memory-mapped
        try:
            self.index = faiss.read_index("image_index.faiss", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        except Exception as e:
            print(f"Index loading error: {e}")
            self.index = None

        # Minimal CSV loading
        try:
            self.valid_image_links = pd.read_csv(
                "valid_image_links.csv",
                usecols=['image_url'],
                dtype={'image_url': str},
                low_memory=True
            )['image_url'].tolist()
        except Exception as e:
            print(f"CSV loading error: {e}")
            self.valid_image_links = []

    def encode_image(self, image):
        """Extremely low-memory image encoding"""
        # Minimize image size
        image = image.resize((32, 32))  # Smallest possible resize

        query_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Ultra-low precision
            query_features = self.model.encode_image(query_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_features = query_features.float().cpu().numpy().astype(np.float16)

        return query_features

    def find_similar_images(self, image):
        """Minimal memory similar image search"""
        if self.index is None or not self.valid_image_links:
            return []

        try:
            query_features = self.encode_image(image)
            distances, indices = self.index.search(query_features, self.max_similar)

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
            # Aggressive cleanup
            torch.cuda.empty_cache()
            gc.collect()

# Global searcher to reduce repeated loading
global_searcher = MemoryEfficientImageSearcher()
global_searcher.load_resources()

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

    @app.route('/find-similar', methods=['POST'])
    def find_similar():
        # Check for uploaded image
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        try:
            # Process uploaded image
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            similar_images = global_searcher.find_similar_images(img)

            return jsonify({
                "similar_images": similar_images,
                "count": len(similar_images)
            }), 200

        except Exception as e:
            print(f"Processing error: {e}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    @app.route("/", methods=["GET"])
    def health_check():
        return jsonify({"status": "running"}), 200

    return app

# Application initialization
app = create_app()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        threaded=False
    )
