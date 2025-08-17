# What you're about to do:
# We will load the trained IVF index and use it to perform a
# lightning-fast search for a given text query.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss

# --- 1. Setup: Load model and define device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print(f"Using device: {device}")

# --- 2. Load the IVF FAISS Index and Paths ---
try:
    # Load the IVF FAISS index
    index = faiss.read_index("gallery_ivf.index")
    
    # Set the nprobe parameter
    # This controls the speed/accuracy trade-off.
    index.nprobe = 16 
    
    # Load the image paths
    with open("image_paths.txt", "r", encoding="utf-8") as f:
        all_image_paths = [line.strip() for line in f.readlines()]
        
    print(f"Successfully loaded IVF index with {index.ntotal} vectors.")

except Exception as e:
    print(f"Error loading files: {e}")
    print("Please run the 'process_images.py' script first to generate the index.")
    exit()

# --- 3. Define and Process the Search Query ---
search_query = "a person smiling"
print(f"\nSearching for: '{search_query}'")

with torch.no_grad():
    # Process the text query to get its embedding
    text_inputs = processor(text=search_query, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)

# Convert the query tensor to a numpy array for FAISS
query_vector = text_features_norm.cpu().numpy().astype('float32')

# --- 4. Search the FAISS Index ---
top_k = 5
distances, indices = index.search(query_vector, k=top_k)

# --- 5. Display the Top Matches ---
print(f"\n--- Top {top_k} Matches ---")
for i in range(top_k):
    image_index = indices[0][i]
    # Note: distance is L2, not cosine. Smaller is better.
    score = distances[0][i]
    image_path = all_image_paths[image_index]
    print(f"Match {i+1} | L2 Distance: {score:.4f} | Path: {image_path}")