import pandas as pd # type: ignore

# Example labeled data
data = [
    {"image_name": "plant_1.png", "mask_name": "plant_1_mask_0.png", "type": "leaf"},
    {"image_name": "plant_1.png", "mask_name": "plant_1_mask_1.png", "type": "stem"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/labels.csv", index=False)
print("Labels CSV created successfully!")