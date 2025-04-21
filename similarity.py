import pandas as pd
import numpy as np
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms

# Load the data
df = pd.read_csv(r'output\data_with_embeddings.csv')

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess query image
query_image = preprocess(Image.open(r'C:\Users\vaibh\Downloads\Black_White-RetroGeometric-BAS00707010-3.webp')).unsqueeze(0).to(device)

# Encode the query image
with torch.no_grad():
    query_embedding = model.encode_image(query_image).cpu().numpy()

# Prepare database embeddings
embeddings = df['embedding'].apply(lambda x: eval(x)).tolist()
embeddings = torch.tensor(embeddings).numpy()
embeddings = np.vstack([emb.squeeze() for emb in embeddings])  # <=== THIS IS IMPORTANT
# Calculate cosine similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Sort by most similar
df['similarity'] = similarities
df_sorted = df.sort_values(by='similarity', ascending=False)

# Print top 5 results
print("\nTop 5 Similar Products:\n")
for idx, row in df_sorted.head(5).iterrows():
    if 'title' in row:
        print(f"Title: {row['title']}")
    print(f"Image URL: {row['image_url']}")
    print(f"Color: {row['color']}")
    print(f"Material: {row['material']}")
    print(f"Style: {row['style']}")
    print(f"Similarity Score: {row['similarity']:.4f}")
    print("-" * 50)
