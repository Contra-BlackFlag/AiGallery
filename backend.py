# What you're about to do:
# This is the complete, final, and self-contained backend script.
# It automatically builds the search index if needed and then launches the API server.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import os
import glob
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Change this to the path of your image folder!
IMAGE_FOLDER = r"Data"
INDEX_FILE = "gallery_ivf.index"
PATHS_FILE = "image_paths.txt"
MODEL_NAME = "openai/clip-vit-base-patch32"

# --- 1. Indexing Function ---
def create_index_if_needed():
    """Checks if the index exists. If not, it creates it."""
    if os.path.exists(INDEX_FILE) and os.path.exists(PATHS_FILE):
        print("Index files found. Skipping creation.")
        return

    print("Index not found. Starting the indexing process...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))

    if not image_paths:
        print(f"Error: No images found in '{IMAGE_FOLDER}'. Please check the path.")
        return

    print(f"Found {len(image_paths)} images. Generating embeddings...")
    
    all_image_features = []
    for path in tqdm(image_paths, desc="Processing Images"):
        try:
            raw_image = Image.open(path).convert("RGB")
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            all_image_features.append(F.normalize(image_features, p=2, dim=-1).cpu().numpy())
        except Exception as e:
            print(f"\nError processing {path}: {e}")

    if not all_image_features:
        print("Could not generate any embeddings.")
        return

    features_array = np.vstack(all_image_features).astype('float32')
    d = features_array.shape[1]
    
    n_partitions = min(1024, len(all_image_features) // 4)
    if n_partitions == 0: n_partitions = 1
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_partitions)
    
    print("Training the FAISS index...")
    index.train(features_array)
    index.add(features_array)
    
    faiss.write_index(index, INDEX_FILE)
    with open(PATHS_FILE, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")
            
    print("--- Indexing complete! ---")

# --- 2. Main Server Setup ---
create_index_if_needed()

print("Server is starting up...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

try:
    index = faiss.read_index(INDEX_FILE)
    index.nprobe = 16
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        all_image_paths = [line.strip() for line in f.readlines()]
    print(f"Successfully loaded IVF index with {index.ntotal} vectors.")
except Exception as e:
    print(f"FATAL: Could not load index or paths file: {e}")
    exit()

app = Flask(__name__)
print("--- Server is now ready to accept requests ---")

# --- 3. API Endpoints ---
@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"status": "ok", "message": "CLIP API server is running."})

# --- THIS IS THE CORRECTED SEARCH FUNCTION ---
@app.route('/search', methods=['GET'])
def search():
    query_text = request.args.get('query')
    if not query_text:
        return jsonify({"error": "Query parameter is missing"}), 400

    print(f"Received search query: '{query_text}'")

    with torch.no_grad():
        text_inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features_norm = F.normalize(text_features, p=2, dim=-1)
    
    query_vector = text_features_norm.cpu().numpy().astype('float32')

    top_k = 10
    distances, indices = index.search(query_vector, k=top_k)

    results = []
    for i in range(top_k):
        image_index = indices[0][i]
        score = distances[0][i]
        filename = os.path.basename(all_image_paths[image_index])
        path_url = f"http://{request.host.split(':')[0]}:5000/images/{filename}"
        results.append({"path": path_url, "score": float(score)})
        
    return jsonify(results)

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

# --- 4. Start the Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)