import cv2


plate_path = "data/processed/license_plate.png"

plate = cv2.imread(plate_path)

if plate is None:
    raise FileNotFoundError(f"Could not load: {plate_path}")


# Convert to grayscale
gray = cv2.cvtColor(
    plate,
    cv2.COLOR_BGR2GRAY
)


# Slight blur
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# --------------------------------
# 1. Fixed threshold
# --------------------------------

_, fixed = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)


# --------------------------------
# 2. Otsu threshold
# --------------------------------

otsu_value, otsu = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# --------------------------------
# 3. Adaptive threshold
# --------------------------------

adaptive = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)


# Save results
cv2.imwrite(
    "data/processed/threshold_fixed.png",
    fixed
)

cv2.imwrite(
    "data/processed/threshold_otsu.png",
    otsu
)

cv2.imwrite(
    "data/processed/threshold_adaptive.png",
    adaptive
)


print("Threshold comparison complete!")
print("Fixed threshold: 127")
print("Otsu selected:", otsu_value)
