import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

# Initialize SAM
def initialize_sam(model_path="models/sam/sam_vit_h_4b8939.pth", model_type="vit_h"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        crop_n_layers=1,
    )
    return mask_generator

# Generate masks for all images
def generate_masks(image_dir, output_dir):
    mask_generator = initialize_sam()
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = mask_generator.generate(image)
        
        # Save masks
        for i, mask in enumerate(masks):
            mask_image = (mask["segmentation"] * 255).astype(np.uint8)
            mask_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask_{i}.png")
            cv2.imwrite(mask_path, mask_image)

# Run
generate_masks("data/raw_images", "data/masks")