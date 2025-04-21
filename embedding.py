import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load your furniture metadata CSV
df = pd.read_csv(r'output\data.csv')

# Prepare storage for embeddings
image_embeddings = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image = preprocess(Image.open(row['image_url'])).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_embeddings.append(image_features.cpu().numpy())
    except Exception as e:
        print(f"Error processing {row['image_url']}: {e}")
        image_embeddings.append(None)



# Save embeddings if needed
import numpy as np
np.save('output/furniture_image_embeddings.npy', image_embeddings)