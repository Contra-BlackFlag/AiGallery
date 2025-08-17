# What you're about to do:
# We will scan a folder for images, generate embeddings, and then use FAISS
# to build and save a fast, partitioned (IVF) search index.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import faiss

# --- 1. Setup: Load model and define device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print(f"Using device: {device}")

# --- 2. Point to your image folder ---
# IMPORTANT: Change this to the path of your image folder!
image_folder = r"Data"
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

print(f"Found {len(image_paths)} images in '{image_folder}'.")

# --- 3. Loop, Process, and Store Embeddings ---
all_image_features = []

for image_path in tqdm(image_paths, desc="Processing Images"):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize and move to CPU before storing
        all_image_features.append(F.normalize(image_features, p=2, dim=-1).cpu().numpy())
    except Exception as e:
        print(f"\nError processing {image_path}: {e}")

# --- 4. Build, Train, and Save the IVF-FAISS Index ---
if all_image_features:
    # Stack all features into a single numpy array
    features_array = np.vstack(all_image_features).astype('float32')
    
    # Get the dimension of the embeddings
    d = features_array.shape[1]
    
    # Create a partitioned IVF index
    n_partitions = min(1024, len(all_image_features) // 4) # Heuristic for n_partitions
    if n_partitions == 0: n_partitions = 1 # Edge case for very few images
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_partitions)
    
    # Train the index on the data
    print("Training the FAISS index...")
    index.train(features_array)
    print("Training complete.")
    
    # Add the embeddings to the index
    index.add(features_array)
    
    print(f"Total embeddings in index: {index.ntotal}")
    
    # Save the index and paths
    faiss.write_index(index, "gallery_ivf.index")
    with open("image_paths.txt", "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")
            
    print("IVF FAISS index and paths file saved successfully!")
else:
    print("No images were processed.")