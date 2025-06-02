"""
Script to probe image resolutions in the original ISIC 2019 dataset.
"""

import os
import pandas as pd
from collections import defaultdict
from PIL import Image
import numpy as np
from tabulate import tabulate

def main():
    # Dataset paths
    data_dir = "/oak/stanford/groups/roxanad/ISIC_2019_Training_Input"
    label_file = "/oak/stanford/groups/roxanad/ISIC_2019_Training_GroundTruth.csv"
    
    # Load labels
    df = pd.read_csv(label_file)
    total_images = len(df)
    
    print(f"Total images in dataset: {total_images}")
    
    # Dictionary to store resolutions
    resolutions = defaultdict(int)
    
    # Sample size for detailed analysis (to avoid processing all images)
    sample_size = min(1000, total_images)
    sampled_indices = np.random.choice(total_images, sample_size, replace=False)
    
    # Collect resolutions
    for idx in sampled_indices:
        img_name = df.iloc[idx]['image'] + '.jpg'
        img_path = os.path.join(data_dir, img_name)
        try:
            with Image.open(img_path) as image:
                width, height = image.size
                resolutions[(width, height)] += 1
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    # Prepare data for tabular display
    table_data = []
    for (width, height), count in sorted(resolutions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sample_size) * 100
        table_data.append([f"{width}x{height}", count, f"{percentage:.1f}%"])
    
    # Print results in a nice table
    print("\nResolution Distribution:")
    print("-----------------------")
    print(tabulate(
        table_data,
        headers=["Resolution", "Count", "Percentage"],
        tablefmt="grid"
    ))

if __name__ == "__main__":
    main() 