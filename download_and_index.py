import pandas as pd
import requests
from PIL import Image
import os
import torch
import clip
import faiss
import numpy as np

def download_and_index_images(excel_file):
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Read the Excel file
    data = pd.read_excel(excel_file)

    # Specify the column containing image links
    image_links = data['IMAGE LINK']

    # Directory to save downloaded images
    download_dir = "downloaded_images"
    os.makedirs(download_dir, exist_ok=True)

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
    faiss.write_index(index, "image_index.faiss")
    pd.DataFrame({'image_url': valid_image_links}).to_csv("valid_image_links.csv", index=False)

    print("Indexing complete. You can now search the FAISS index.")

if __name__ == "__main__":
    download_and_index_images("mexbidimagesdata.xlsx")