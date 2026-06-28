import pandas as pd
import numpy as np
import os

def preprocess_myntra_data(input_file, output_file, sample_size=100000):
    print("Loading dataset (this might take a minute for 1.4GB)...")
    # Load the Kaggle dataset
    df = pd.read_csv(input_file, on_bad_lines='skip')
    
    print("Renaming columns...")
    # 1. Rename columns to match what the Streamlit app expects
    df = df.rename(columns={
        'id': 'product_id',
        'name': 'product_name',
        'seller': 'brand',
        'mrp': 'original_price',
        'price': 'discounted_price',
        'ratingTotal': 'num_reviews',
        'img': 'image_url',
        'purl': 'product_url'
    })
    
    print("Extracting categories...")
    # 2. Derive the 'category' column based on keywords in the product name
    def get_category(name):
        name = str(name).lower()
        if 'women' in name or 'girls' in name:
            return 'Women'
        elif 'men' in name or 'boys' in name:
            return 'Men'
        elif 'kids' in name or 'baby' in name:
            return 'Kids'
        else:
            return 'Unisex'
            
    df['category'] = df['product_name'].apply(get_category)
    
    print("Cleaning image URLs and discount percentage...")
    # 3. Clean the image URL (Kaggle dataset has multiple images separated by ';', we just want the first one)
    df['image_url'] = df['image_url'].astype(str).str.split(';').str[0]

    # 4. Clean the discount column (ensure it's a percentage)
    # The safest way is to recalculate it using MRP and Selling Price
    df['discount_pct'] = ((df['original_price'] - df['discounted_price']) / df['original_price']) * 100
    df['discount_pct'] = df['discount_pct'].fillna(0).clip(lower=0, upper=100).round(2)
    
    print("Dropping unused columns...")
    # 5. Drop columns the app doesn't need to save space
    columns_to_keep = ['product_id', 'product_name', 'brand', 'category', 
                       'original_price', 'discount_pct', 'discounted_price', 
                       'rating', 'num_reviews', 'image_url', 'product_url']
    df = df[columns_to_keep]
    
    print("Cleaning up bad data...")
    # 6. Clean up bad data (drop rows with missing essential info or zero MRP)
    df = df.dropna(subset=['product_id', 'product_name', 'original_price', 'image_url', 'product_url'])
    df = df[df['original_price'] > 0]
    
    print(f"Sampling {sample_size} rows...")
    # 7. Sample the data so the Streamlit app runs fast during the interview
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        
    print(f"Saving to {output_file}...")
    # 8. Save the cleaned dataset
    df.to_csv(output_file, index=False)
    print("Done! Your data is ready for the app.")

if __name__ == "__main__":
    # Adjust this path if the CSV is located somewhere else
    input_csv_path = "myntra202305041052.csv/myntra202305041052.csv"
    output_csv_path = "cleaned_myntra_data.csv"
    
    if os.path.exists(input_csv_path):
        preprocess_myntra_data(input_csv_path, output_csv_path)
    else:
        print(f"Error: Could not find the input file at '{input_csv_path}'. Please check the path.")
