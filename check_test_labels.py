from pathlib import Path
import random

from PIL import Image, ImageDraw


DATASET = Path("data/car_test")
IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"

OUTPUT_DIR = Path("data/test_sanity_check")

NUM_POSITIVE_SAMPLES = 10
NUM_NEGATIVE_SAMPLES = 5

random.seed(42)


# ------------------------------------------------------------
# Prepare output folder
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in OUTPUT_DIR.iterdir():
    if p.is_file():
        p.unlink()


# ------------------------------------------------------------
# Integrity check
# ------------------------------------------------------------

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in image_extensions
])

missing_labels = []
invalid_rows = []
class_ids = set()

positive_images = []
negative_images = []

total_boxes = 0


for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        missing_labels.append(image_path.name)
        continue

    lines = [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    if lines:
        positive_images.append(image_path)
    else:
        negative_images.append(image_path)

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) != 5:
            invalid_rows.append(
                (label_path.name, line_number, line)
            )
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            invalid_rows.append(
                (label_path.name, line_number, line)
            )
            continue

        class_ids.add(class_id)

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            invalid_rows.append(
                (label_path.name, line_number, line)
            )
            continue

        total_boxes += 1


print("=== Test Dataset Integrity Check ===")
print("Images:", len(images))
print("Missing labels:", len(missing_labels))
print("Invalid label rows:", len(invalid_rows))
print("Class IDs:", sorted(class_ids))
print("Positive images:", len(positive_images))
print("Negative images:", len(negative_images))
print("Total car boxes:", total_boxes)


# ------------------------------------------------------------
# Visualize positive samples
# ------------------------------------------------------------

positive_samples = random.sample(
    positive_images,
    min(NUM_POSITIVE_SAMPLES, len(positive_images))
)

for image_path in positive_samples:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    draw = ImageDraw.Draw(image)

    for line in label_path.read_text().splitlines():

        if not line.strip():
            continue

        _, xc, yc, bw, bh = map(float, line.split())

        xc *= w
        yc *= h
        bw *= w
        bh *= h

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 12)),
            "car",
            fill="red"
        )

    image.save(
        OUTPUT_DIR / f"positive_{image_path.name}"
    )


# ------------------------------------------------------------
# Copy negative samples for visual inspection
# ------------------------------------------------------------

negative_samples = random.sample(
    negative_images,
    min(NUM_NEGATIVE_SAMPLES, len(negative_images))
)

for image_path in negative_samples:

    image = Image.open(image_path).convert("RGB")

    image.save(
        OUTPUT_DIR / f"negative_{image_path.name}"
    )


print()
print("Visualization saved to:")
print(OUTPUT_DIR)