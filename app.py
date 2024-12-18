import os
import io
import gc
import sys
import torch
import faiss
import clip
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add memory logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraLightImageSearcher:
    def __init__(self, max_similar=3):
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self.index = None
        self.valid_image_links = None
        self.max_similar = max_similar

    def load_resources(self):
        """Extreme memory-conscious resource loading"""
        try:
            # Use smallest possible model
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            # Verify index file exists and is readable
            if not os.path.exists("image_index.faiss"):
                logger.error("FAISS index file not found!")
                return False

            # Minimal index loading
            self.index = faiss.read_index("image_index.faiss", faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            
            # Ultra-minimal CSV loading
            if not os.path.exists("valid_image_links.csv"):
                logger.error("Image links CSV not found!")
                return False

            self.valid_image_links = pd.read_csv(
                "valid_image_links.csv",
                usecols=['image_url'],
                dtype={'image_url': str},
                nrows=1000,  # Limit rows to reduce memory
                low_memory=True
            )['image_url'].tolist()

            logger.info(f"Loaded {len(self.valid_image_links)} image links")
            return True

        except Exception as e:
            logger.error(f"Resource loading error: {e}")
            return False

    def encode_image(self, image):
        """Minimal memory image encoding"""
        try:
            # Extreme resize
            image = image.resize((16, 16))  # Tiny image

            query_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                query_features = self.model.encode_image(query_input)
                query_features /= query_features.norm(dim=-1, keepdim=True)
                query_features = query_features.float().cpu().numpy().astype(np.float16)

            return query_features

        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            return None

    def find_similar_images(self, image):
        """Absolute minimal memory similar image search"""
        try:
            query_features = self.encode_image(image)
            
            if query_features is None or self.index is None:
                return []

            distances, indices = self.index.search(query_features, self.max_similar)

            similar_images = [
                self.valid_image_links[idx]
                for idx in indices[0]
                if 0 <= idx < len(self.valid_image_links)
            ]

            return similar_images

        except Exception as e:
            logger.error(f"Similar image search error: {e}")
            return []

        finally:
            # Aggressive cleanup
            torch.cuda.empty_cache()
            gc.collect()

# Global searcher with minimal initialization
def create_safe_searcher():
    searcher = UltraLightImageSearcher()
    if not searcher.load_resources():
        logger.error("Failed to load resources. Exiting.")
        sys.exit(1)
    return searcher

# Create searcher once
global_searcher = create_safe_searcher()

def create_app():
    app = Flask(__name__)
    
    CORS(app, resources={
        r"/find-similar": {
            "origins": ["*"],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })

    @app.route('/find-similar', methods=['POST'])
    def find_similar():
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            similar_images = global_searcher.find_similar_images(img)

            return jsonify({
                "similar_images": similar_images,
                "count": len(similar_images)
            }), 200

        except Exception as e:
            logger.error(f"Processing error: {e}")
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
