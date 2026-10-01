import cv2


# 1. Load cropped license plate
plate_path = "data/processed/license_plate.png"

plate = cv2.imread(plate_path)

if plate is None:
    raise FileNotFoundError(f"Could not load: {plate_path}")

print("Original plate shape:", plate.shape)


# 2. Convert to grayscale
gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

print("Grayscale shape:", gray.shape)


# 3. Apply Gaussian blur
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# 4. Apply Otsu thresholding
_, threshold = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# 5. Detect edges
edges = cv2.Canny(
    blurred,
    50,
    150
)


# 6. Save results
cv2.imwrite(
    "data/processed/plate_gray.png",
    gray
)

cv2.imwrite(
    "data/processed/plate_blurred.png",
    blurred
)

cv2.imwrite(
    "data/processed/plate_threshold.png",
    threshold
)

cv2.imwrite(
    "data/processed/plate_edges.png",
    edges
)

print("Preprocessing complete!")
