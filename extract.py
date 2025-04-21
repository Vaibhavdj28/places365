import requests
import pandas as pd
import os
import time

BASE_URL = "https://lofy.in"

# Target collections
collections = [
    "sofas",
    "beds",
    "center-table-coffee-table",
    "dining-table",
    "chair",
    "bench"
]

# Storage
all_products = []

def clean_text(text):
    if text:
        return text.strip()
    return ""

def scrape_collection(collection_handle):
    page = 1
    while True:
        url = f"{BASE_URL}/collections/{collection_handle}/products.json?page={page}"
        print(f"Scraping {url}...")
        res = requests.get(url)
        if res.status_code != 200:
            print(f"Failed to load {url}")
            break

        data = res.json()
        products = data.get('products', [])
        if not products:
            print(f"No more products found in {collection_handle}.")
            break

        for prod in products:
            product_name = clean_text(prod['title'])
            image_url = prod['images'][0]['src'] if prod['images'] else ''
            tags = prod.get('tags', [])

            # Metadata extraction
            color = "Unknown"
            material = "Unknown"
            style = "Modern"  # default

            for tag in tags:
                tag_lower = tag.lower()
                if any(c in tag_lower for c in ["black", "white", "grey", "brown", "blue", "beige"]):
                    color = tag.capitalize()
                if any(m in tag_lower for m in ["wood", "metal", "fabric", "leather", "glass"]):
                    material = tag.capitalize()
                if any(s in tag_lower for s in ["modern", "traditional", "contemporary", "minimalist"]):
                    style = tag.capitalize()

            all_products.append({
                "product_name": product_name,
                "image_url": image_url,
                "category": collection_handle,
                "color": color,
                "material": material,
                "style": style
            })

        page += 1
        time.sleep(1)  # be polite

def main():
    os.makedirs('output', exist_ok=True)

    for collection_handle in collections:
        scrape_collection(collection_handle)

    df = pd.DataFrame(all_products)
    df.to_csv("output/furniture_metadata.csv", index=False)
    print("\n✅ Scraping completed. CSV saved to output/furniture_metadata.csv")

if __name__ == "__main__":
    main()
