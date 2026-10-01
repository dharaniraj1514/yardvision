import cv2
import numpy as np


image_path = "data/raw/truck.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

print("Python type:", type(image))
print("Is NumPy array:", isinstance(image, np.ndarray))

print("\nImage information")
print("-----------------")
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum value:", image.min())
print("Maximum value:", image.max())
# Convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("\n--- Grayscale Image ---")
print("Original shape:", image.shape)
print("Grayscale shape:", gray.shape)
print("Grayscale data type:", gray.dtype)

# Save the grayscale image
output_path = "data/processed/truck_gray.png"

cv2.imwrite(output_path, gray)

print("Saved to:", output_path)
# Resize image
resized = cv2.resize(image, (640, 640))

print("\n--- Resized Image ---")
print("Before:", image.shape)
print("After:", resized.shape)

cv2.imwrite("data/processed/truck_resized.png", resized)
# Crop the license plate region
plate_crop = image[620:690, 330:435]

print("\n--- License Plate Crop ---")
print("Original shape:", image.shape)
print("Plate crop shape:", plate_crop.shape)

cv2.imwrite(
    "data/processed/license_plate.png",
    plate_crop
)
