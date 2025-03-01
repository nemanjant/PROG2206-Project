# Plant Segmenta1on and Trait Analysis

Implement a segmentation model to extract plant structures (e.g., leaves, stems) for automated trait
measurement. Utilize SAM to segment plant images into meaningful regions.
Extract features like leaf count, size, and shape for phenotypic analysis. Useful for plant growth studies,
breeding programs, and automated monitoring.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Add to models/sam:
   https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth

## How to use program

1. Store picture of the plant in data/raw_images
2. Run scripts\generate_masks.py
3. Select all relevant masks for stem and leaf (more is better, even numbers)
4. Make changes in labels.csv
5. Run scripts\utils\dataset.py
6. Run scripts\train_classifier.py, where output should be like (support column should have more samples, minimum 40 for better precision)
7. Run scripts\inference.py
8. Check output graphic and data\results\plant_image_features.csv for results and if it's needed make changes:
   - Use bigger sample, more masks
   - Change masks if sample is big enough
   - Select picture with better resolution
