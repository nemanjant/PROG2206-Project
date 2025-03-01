import os
import cv2
import numpy as np
import torch
import joblib
import pandas as pd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Load and preprocess image
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}. The file may be corrupted or in an unsupported format.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Initialize SAM
def initialize_sam(model_path="models/sam/sam_vit_h_4b8939.pth", model_type="vit_h"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,       # Adjust points per side
        pred_iou_thresh=0.9,      # Adjust IoU threshold
        stability_score_thresh=0.92,  # Adjust stability score
        crop_n_layers=1,
    )
    return mask_generator

# Extract features from masks
def extract_features(masks, image):
    features_list = []
    for mask_data in masks:
        mask = mask_data["segmentation"]
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
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# Classify plant parts
def classify_plant_part(features, clf):
    features_df = pd.DataFrame([features])
    return clf.predict(features_df)[0]

# Filter masks by size
def filter_masks(masks, max_stem_area=1000):
    filtered_masks = []
    for mask in masks:
        if mask["area"] < max_stem_area:
            filtered_masks.append(mask)
    return filtered_masks

# Visualize results
def visualize(image, df, masks):
    overlay = image.copy()
    for idx, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        label = df.iloc[idx]["type"]
        print(f"Mask {idx + 1}: Label = {label}, Area = {df.iloc[idx]['area']}, Eccentricity = {df.iloc[idx]['eccentricity']}")
        
        color = [255, 100, 100] if label == "leaf" else [100, 100, 255]  # Red=leaf, Blue=stem
        
        # Apply mask to the overlay
        overlay[mask] = color

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Segmented Leaves (Red) and Stems (Blue)")
    plt.axis("off")
    plt.show()

# Main pipeline
def process_image(image_path, mask_generator, clf):
    image = load_image(image_path)
    print(f"Processing {image_path}...")

    masks = mask_generator.generate(image)
    print(f"Found {len(masks)} plant structures.")

    # Extract features
    features_df = extract_features(masks, image)
    
    # Debug: Print feature names
    print("Feature Names in features_df:", features_df.columns.tolist())
    
    # Classify each segment
    features_df["type"] = [classify_plant_part(features, clf) for _, features in features_df.iterrows()]
    
    # Print classification results
    print("Classification Results:")
    print(features_df[["type"]])
    
    # Save results
    base_name = os.path.basename(image_path).split('.')[0]
    features_df.to_csv(f"data/results/{base_name}_features.csv", index=False)
    
    # Visualize
    visualize(image, features_df, masks)

# Run inference
if __name__ == "__main__":
    mask_generator = initialize_sam()
    clf = joblib.load("models/classifier/leaf_stem_classifier.pth")
    image_path = "data/raw_images/plant_image.jpg"  # Replace with your image path
    process_image(image_path, mask_generator, clf)