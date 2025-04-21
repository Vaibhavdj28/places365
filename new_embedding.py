import pandas as pd
import numpy as np

# Load your original data
df = pd.read_csv(r'output\data.csv')

# Load the saved embeddings
image_embeddings = np.load('output/furniture_image_embeddings.npy', allow_pickle=True)

# Check if lengths match
assert len(df) == len(image_embeddings), "Mismatch between number of rows and embeddings!"

# Add embeddings as a new column
df['embedding'] = image_embeddings.tolist()

# Save to a new CSV (with embeddings inside)
df.to_csv(r'output\data_with_embeddings.csv', index=False)

print("Embeddings successfully added and saved to data_with_embeddings.csv ✅")
