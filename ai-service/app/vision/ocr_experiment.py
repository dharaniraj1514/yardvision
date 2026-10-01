import easyocr
import re
from rapidfuzz.distance import Levenshtein
GROUND_TRUTH = "RJ14GT4976"


# Create OCR reader
# gpu=False because we're currently running on CPU
reader = easyocr.Reader(["en"], gpu=False)


images = {
    "original": "data/processed/license_plate.png",
    "grayscale": "data/processed/plate_gray.png",
    "otsu": "data/processed/threshold_otsu.png",
    "adaptive": "data/processed/threshold_adaptive.png",
}


def clean_text(text):
    """
    Remove spaces and special characters.
    Keep only A-Z and 0-9.
    """
    return re.sub(r"[^A-Z0-9]", "", text.upper())


print("\n===== YardVision OCR Experiment =====\n")


for method, image_path in images.items():

    print(f"Processing: {method}")

    results = reader.readtext(image_path)

    detected_parts = []

    for result in results:
        bounding_box, text, confidence = result

        detected_parts.append(text)

        print(
            f"  Detected: {text} "
            f"| Confidence: {confidence:.2f}"
        )

    raw_text = " ".join(detected_parts)

    cleaned = clean_text(raw_text)
    distance = Levenshtein.distance(
    GROUND_TRUTH,
    cleaned
)

    max_length = max(
    len(GROUND_TRUTH),
    len(cleaned)
)

    accuracy = (
    1 - distance / max_length
) * 100

    print("  Ground truth:", GROUND_TRUTH)
    print("  Edit distance:", distance)
    print(f"  Accuracy: {accuracy:.2f}%")

    print("  Raw text:", raw_text)
    print("  Cleaned :", cleaned)
    print()
