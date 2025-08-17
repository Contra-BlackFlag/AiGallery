# What you're about to do:
# We will load the pre-built FAISS index and use it to perform a
# lightning-fast search for a given text query.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import faiss # Import FAISS

# --- 1. Setup: Load model and define device ---
# (This part is unchanged)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print(f"Using device: {device}")

# --- 2. NEW: Load the FAISS Index and Paths ---
try:
    # Load the FAISS index
    index = faiss.read_index("gallery.index")
    
    # Load the corresponding image paths
    with open("image_paths.txt", "r", encoding="utf-8") as f:
        all_image_paths = [line.strip() for line in f.readlines()]
        
    print(f"Successfully loaded FAISS index with {index.ntotal} vectors.")

except Exception as e:
    print(f"Error loading files: {e}")
    print("Please run the 'process_images.py' script first to generate the index.")
    exit()

# --- 3. Define and Process the Search Query ---
# (This part is unchanged)
search_query = "a person smiling"
print(f"\nSearching for: '{search_query}'")

with torch.no_grad():
    text_inputs = processor(text=search_query, return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)

# Convert the query tensor to a numpy array for FAISS
query_vector = text_features_norm.cpu().numpy()


# --- 4. NEW: Search the FAISS Index ---
top_k = 5
# The search method returns distances and indices
distances, indices = index.search(query_vector, k=top_k)

# --- 5. Display the Top Matches ---
print(f"\n--- Top {top_k} Matches ---")
for i in range(top_k):
    # The indices array is 2D, so we access the first row
    image_index = indices[0][i]
    score = distances[0][i] # L2 distance, smaller is better
    image_path = all_image_paths[image_index]
    print(f"Match {i+1} | Score: {score:.4f} | Path: {image_path}")