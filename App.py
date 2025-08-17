# What you're about to do:
# We will generate all image embeddings and then use FAISS
# to build and save an efficient search index.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import faiss  # Import FAISS

# --- 1. Setup: Load model and define device ---
# (This part is unchanged)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
print(f"Using device: {device}")

# --- 2. Point to your image folder ---
# (This part is unchanged)
image_folder = r"Data"
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
print(f"Found {len(image_paths)} images in '{image_folder}'.")

# --- 3. Loop, Process, and Store Embeddings ---
# (This part is unchanged)
all_image_features = []
for image_path in tqdm(image_paths, desc="Processing Images"):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        all_image_features.append(F.normalize(image_features, p=2, dim=-1).cpu().numpy())
    except Exception as e:
        print(f"\nError processing {image_path}: {e}")

# --- 4. NEW: Build and Save the FAISS Index ---
if all_image_features:
    # Stack all features into a single numpy array
    features_array = np.vstack(all_image_features)
    
    # Get the dimension of the embeddings
    d = features_array.shape[1]
    
    # Create a FAISS index
    # IndexFlatL2 is a simple index that performs an exact search
    index = faiss.IndexFlatL2(d)
    
    # Add the embeddings to the index
    index.add(features_array)
    
    print(f"\nTotal embeddings in index: {index.ntotal}")
    
    # Save the index to a file
    faiss.write_index(index, "gallery.index")
    
    # Save the paths file as before
    with open("image_paths.txt", "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(path + "\n")
            
    print("FAISS index and paths file saved successfully!")
else:
    print("No images were processed.")