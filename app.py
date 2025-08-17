# What you're about to do:
# This is the complete, final version of the Flask server with all
# endpoints (`/`, `/search`, `/add_image`) correctly structured.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss
from flask import Flask, request, jsonify
from PIL import Image
import os
import threading

# --- 1. ONE-TIME SETUP: Load models and index at startup ---
print("Server is starting up...")

# Create a lock to handle safe concurrent additions to the index
index_lock = threading.Lock()

# Load model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print(f"Using device: {device}")

# Load the FAISS index and paths
try:
    index = faiss.read_index("gallery_ivf.index")
    index.nprobe = 16
    with open("image_paths.txt", "r", encoding="utf-8") as f:
        all_image_paths = [line.strip() for line in f.readlines()]
    print(f"Successfully loaded IVF index with {index.ntotal} vectors.")
except Exception as e:
    print(f"FATAL: Could not load index or paths file: {e}")
    index = None
    all_image_paths = []

# Initialize the Flask application
app = Flask(__name__)
print("--- Server is now ready to accept requests ---")


# --- 2. WELCOME ENDPOINT ---
@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"status": "ok", "message": "CLIP API server is running."})


# --- 3. SEARCH ENDPOINT ---
@app.route('/search', methods=['GET'])
def search():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 500

    query_text = request.args.get('query')
    if not query_text:
        return jsonify({"error": "Query parameter is missing"}), 400

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
        path = all_image_paths[image_index]
        results.append({"path": path, "score": float(score)})
        
    return jsonify(results)

# --- 4. ADD IMAGE ENDPOINT ---
@app.route('/add_image', methods=['POST'])
def add_image():
    if index is None:
        return jsonify({"error": "Index not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        raw_image = Image.open(file.stream).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            image_features_norm = F.normalize(image_features, p=2, dim=-1)
        
        new_vector = image_features_norm.cpu().numpy().astype('float32')
        
        with index_lock:
            index.add(new_vector)
            all_image_paths.append(file.filename) 
        
        print(f"Successfully added '{file.filename}'. Index now contains {index.ntotal} vectors.")
        
        return jsonify({"success": True, "message": f"'{file.filename}' added to the index."})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- 5. START THE SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)