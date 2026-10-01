import cv2
import easyocr
import re

from rapidfuzz.distance import Levenshtein


GROUND_TRUTH = "RJ14GT4976"

IMAGE_PATH = "data/processed/license_plate.png"

reader = easyocr.Reader(["en"], gpu=False)


def clean_text(text):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


def calculate_score(prediction):

    distance = Levenshtein.distance(
        GROUND_TRUTH,
        prediction
    )

    max_length = max(
        len(GROUND_TRUTH),
        len(prediction)
    )

    if max_length == 0:
        return distance, 0.0

    score = (
        1 - distance / max_length
    ) * 100

    return distance, score


# -------------------------
# Load plate
# -------------------------

plate = cv2.imread(IMAGE_PATH)

if plate is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}"
    )


# -------------------------
# Convert to grayscale
# -------------------------

gray = cv2.cvtColor(
    plate,
    cv2.COLOR_BGR2GRAY
)


# -------------------------
# CLAHE
# -------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)


# -------------------------
# Upscale enhanced image
# -------------------------

enhanced_upscaled = cv2.resize(
    enhanced,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC
)


cv2.imwrite(
    "data/processed/plate_clahe.png",
    enhanced
)

cv2.imwrite(
    "data/processed/plate_clahe_upscaled.png",
    enhanced_upscaled
)


# -------------------------
# OCR experiment
# -------------------------

images = {
    "grayscale": gray,
    "CLAHE": enhanced,
    "CLAHE + 3x": enhanced_upscaled
}


print(
    "\n===== CLAHE OCR Experiment =====\n"
)


for method, image in images.items():

    results = reader.readtext(image)

    detected_parts = []

    print(f"Processing: {method}")

    for _, text, confidence in results:

        detected_parts.append(text)

        print(
            f"  Detected: {text}"
            f" | Confidence: {confidence:.2f}"
        )

    raw = " ".join(detected_parts)

    cleaned = clean_text(raw)

    distance, score = calculate_score(
        cleaned
    )

    print("  Raw:", raw)
    print("  Cleaned:", cleaned)
    print("  Ground truth:", GROUND_TRUTH)
    print("  Edit distance:", distance)
    print(
        f"  Similarity score: {score:.2f}%"
    )

    print()
