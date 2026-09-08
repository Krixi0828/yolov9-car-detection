from pathlib import Path
import random

from PIL import Image, ImageDraw


DATASET = Path("data/car_dataset_exp3")
IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

OUTPUT_DIR = Path("data/exp3_sanity_check")

NUM_SAMPLES = 12
random.seed(42)


# ------------------------------------------------------------
# Prepare output folder
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for old_file in OUTPUT_DIR.glob("*"):
    if old_file.is_file():
        old_file.unlink()


# ------------------------------------------------------------
# Dataset integrity check
# ------------------------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

images = [
    p
    for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in image_extensions
]

missing_labels = []
invalid_rows = []
class_ids = set()
total_boxes = 0


for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        missing_labels.append(image_path.name)
        continue

    with open(label_path) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                invalid_rows.append(
                    (
                        label_path.name,
                        line_number,
                        "expected 5 values",
                        line,
                    )
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
                    (
                        label_path.name,
                        line_number,
                        "invalid number",
                        line,
                    )
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
                    (
                        label_path.name,
                        line_number,
                        "coordinate outside YOLO range",
                        line,
                    )
                )
                continue

            total_boxes += 1


print("=== Dataset Integrity Check ===")
print("Train images:", len(images))
print("Missing labels:", len(missing_labels))
print("Invalid label rows:", len(invalid_rows))
print("Class IDs:", sorted(class_ids))
print("Total bounding boxes:", total_boxes)


if missing_labels:
    print("\nFirst missing labels:")
    for item in missing_labels[:10]:
        print(item)


if invalid_rows:
    print("\nFirst invalid rows:")
    for item in invalid_rows[:10]:
        print(item)


# ------------------------------------------------------------
# Select UA-DETRAC images
# ------------------------------------------------------------

detrac_images = [
    p
    for p in images
    if p.name.startswith("detrac_")
]

samples = random.sample(
    detrac_images,
    min(NUM_SAMPLES, len(detrac_images)),
)


# ------------------------------------------------------------
# Draw YOLO bounding boxes
# ------------------------------------------------------------

for image_path in samples:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")

    image_width, image_height = image.size

    draw = ImageDraw.Draw(image)

    with open(label_path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            class_id, x_center, y_center, width, height = map(
                float,
                line.split(),
            )

            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2

            x2 = x_center + width / 2
            y2 = y_center + height / 2

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3,
            )

            draw.text(
                (x1, max(0, y1 - 12)),
                "car",
                fill="red",
            )

    output_path = OUTPUT_DIR / image_path.name

    image.save(
        output_path,
        quality=90,
    )


print()
print(
    f"Saved {len(samples)} visualization images to:"
)
print(OUTPUT_DIR)