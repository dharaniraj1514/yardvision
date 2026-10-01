import cv2
import easyocr
import re

from rapidfuzz.distance import Levenshtein


GROUND_TRUTH = "RJ14GT4976"

IMAGE_PATH = "data/processed/license_plate.png"

reader = easyocr.Reader(["en"], gpu=False)


def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


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


plate = cv2.imread(IMAGE_PATH)

if plate is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}"
    )


print("Original shape:", plate.shape)


methods = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


for name, interpolation in methods.items():

    # Enlarge image by 3x
    upscaled = cv2.resize(
        plate,
        None,
        fx=3,
        fy=3,
        interpolation=interpolation
    )

    output_path = (
        f"data/processed/"
        f"plate_upscaled_{name}.png"
    )

    cv2.imwrite(output_path, upscaled)

    results = reader.readtext(upscaled)

    detected_parts = []

    for _, text, confidence in results:

        detected_parts.append(text)

        print(
            f"{name}: detected '{text}' "
            f"(confidence={confidence:.2f})"
        )

    raw_text = " ".join(detected_parts)

    cleaned = clean_text(raw_text)

    distance, score = calculate_score(cleaned)

    print("  Shape:", upscaled.shape)
    print("  Raw:", raw_text)
    print("  Cleaned:", cleaned)
    print("  Edit distance:", distance)
    print(f"  Similarity score: {score:.2f}%")
    print()
