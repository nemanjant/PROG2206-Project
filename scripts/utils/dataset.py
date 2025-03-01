import os
import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
import pandas as pd

class PlantDataset:
    def __init__(self, image_dir, mask_dir, labels_csv):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = pd.read_csv(labels_csv)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx]["image_name"]
        mask_name = self.labels.iloc[idx]["mask_name"]
        class_label = self.labels.iloc[idx]["type"]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        features = self.extract_features(mask, image)
        features["type"] = class_label
        return features
    
    def extract_features(self, mask, image):
        """Extract features from a binary mask and image."""
        props = regionprops(mask.astype(int))[0]
        
        # Basic shape features
        area = props.area
        perimeter = props.perimeter
        eccentricity = props.eccentricity
        solidity = props.solidity
        extent = props.extent
        convex_area = props.convex_area
        
        # Additional features
        aspect_ratio = props.major_axis_length / props.minor_axis_length
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        
        # Texture features (example: GLCM contrast)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        
        features = {
            "area": area,
            "perimeter": perimeter,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "extent": extent,
            "convex_area": convex_area,
            "aspect_ratio": aspect_ratio,
            "compactness": compactness,
            "contrast": contrast,
        }
        return features

# Create a new CSV with features
def create_feature_csv(labels_csv, output_csv):
    dataset = PlantDataset("data/raw_images", "data/masks", labels_csv)
    features_list = []
    
    for idx in range(len(dataset)):
        features = dataset[idx]
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Feature CSV saved to {output_csv}")

# Run
create_feature_csv("data/labels.csv", "data/labels_with_features.csv")