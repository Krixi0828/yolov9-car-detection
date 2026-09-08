from pathlib import Path
from collections import defaultdict
from io import BytesIO
import shutil
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

DETRAC_RAW = DATA_DIR / "UA-DETRAC_raw"
DETRAC_ZIP = DETRAC_RAW / "DETRAC-Images.zip"
DETRAC_XML_DIR = DETRAC_RAW / "DETRAC-Train-Annotations-XML"

CUSTOM_DATASET = DATA_DIR / "car_dataset"

OUTPUT_DATASET = DATA_DIR / "car_dataset_exp3"
OUTPUT_YAML = DATA_DIR / "car_exp3.yaml"


# ============================================================
# Experiment settings
# ============================================================

NUM_DETRAC_IMAGES = 500


# ============================================================
# Utility functions
# ============================================================

def evenly_sample(items, n):
    """
    從同一個 sequence 中平均取 n 個 frames，
    避免全部取到連續且非常相似的 frames。
    """

    if n >= len(items):
        return items

    if n == 1:
        return [items[len(items) // 2]]

    indices = [
        round(i * (len(items) - 1) / (n - 1))
        for i in range(n)
    ]

    return [items[i] for i in indices]


def copy_custom_split(split):
    """
    將原本 car_dataset 的 train / val 複製到 Experiment 3。

    檔名前加 custom_，避免和 UA-DETRAC 名稱衝突。
    """

    src_images = CUSTOM_DATASET / "images" / split
    src_labels = CUSTOM_DATASET / "labels" / split

    dst_images = OUTPUT_DATASET / "images" / split
    dst_labels = OUTPUT_DATASET / "labels" / split

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    count = 0

    for image_path in sorted(src_images.iterdir()):

        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in image_extensions
        ):
            continue

        label_path = src_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for custom image: {image_path}"
            )

        new_image_name = f"custom_{image_path.name}"
        new_label_name = f"custom_{image_path.stem}.txt"

        shutil.copy2(
            image_path,
            dst_images / new_image_name
        )

        shutil.copy2(
            label_path,
            dst_labels / new_label_name
        )

        count += 1

    return count


# ============================================================
# Prepare output directory
# ============================================================

if OUTPUT_DATASET.exists():
    print(f"Removing old dataset: {OUTPUT_DATASET}")
    shutil.rmtree(OUTPUT_DATASET)

for split in ["train", "val"]:

    (OUTPUT_DATASET / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DATASET / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# Read official UA-DETRAC annotations
# ============================================================

print("Reading UA-DETRAC annotations...")

sequence_frames = defaultdict(list)

xml_files = sorted(
    DETRAC_XML_DIR.glob("MVI_*.xml")
)

if not xml_files:
    raise FileNotFoundError(
        f"No XML files found in {DETRAC_XML_DIR}"
    )


for xml_path in xml_files:

    sequence = xml_path.stem

    root = ET.parse(xml_path).getroot()

    for frame in root.findall("frame"):

        frame_num = int(frame.get("num"))

        car_boxes = []

        for target in frame.findall(
            "./target_list/target"
        ):

            box = target.find("box")
            attribute = target.find("attribute")

            if box is None or attribute is None:
                continue

            vehicle_type = attribute.get(
                "vehicle_type"
            )

            # Experiment 3 only learns:
            #
            # 0 = car
            #
            # bus / van / others are NOT car.
            if vehicle_type != "car":
                continue

            left = float(box.get("left"))
            top = float(box.get("top"))
            width = float(box.get("width"))
            height = float(box.get("height"))

            if width <= 0 or height <= 0:
                continue

            car_boxes.append(
                (left, top, width, height)
            )

        # Only select frames containing at least one car.
        if car_boxes:

            sequence_frames[sequence].append(
                {
                    "frame_num": frame_num,
                    "boxes": car_boxes,
                }
            )


print(
    f"Sequences containing car: "
    f"{len(sequence_frames)}"
)

total_candidate_frames = sum(
    len(frames)
    for frames in sequence_frames.values()
)

print(
    f"Candidate frames containing car: "
    f"{total_candidate_frames}"
)


# ============================================================
# Allocate 500 images across sequences
# ============================================================

sequences = sorted(sequence_frames.keys())

base = NUM_DETRAC_IMAGES // len(sequences)
remainder = NUM_DETRAC_IMAGES % len(sequences)

selected = []


for i, sequence in enumerate(sequences):

    # Example:
    #
    # 500 / 60
    #
    # approximately 8-9 images from each sequence.
    n = base + (1 if i < remainder else 0)

    frames = sequence_frames[sequence]

    chosen_frames = evenly_sample(
        frames,
        n
    )

    for frame in chosen_frames:

        selected.append(
            (
                sequence,
                frame["frame_num"],
                frame["boxes"],
            )
        )


print(
    f"Selected UA-DETRAC frames: "
    f"{len(selected)}"
)


# ============================================================
# Extract ONLY selected images from the 9 GB ZIP
# ============================================================

print()
print("Extracting selected images from DETRAC-Images.zip...")

detrac_car_instances = 0


with zipfile.ZipFile(DETRAC_ZIP, "r") as zf:

    zip_members = set(zf.namelist())

    for index, (sequence, frame_num, boxes) in enumerate(
        selected,
        start=1
    ):

        image_name = f"img{frame_num:05d}.jpg"

        zip_member = (
            f"DETRAC-Images/"
            f"{sequence}/"
            f"{image_name}"
        )

        if zip_member not in zip_members:
            raise FileNotFoundError(
                f"Image not found in ZIP: "
                f"{zip_member}"
            )

        image_bytes = zf.read(zip_member)

        # Read image dimensions without extracting
        # the entire dataset.
        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            image_width, image_height = image.size

        output_stem = (
            f"detrac_{sequence}_"
            f"img{frame_num:05d}"
        )

        output_image = (
            OUTPUT_DATASET
            / "images"
            / "train"
            / f"{output_stem}.jpg"
        )

        output_label = (
            OUTPUT_DATASET
            / "labels"
            / "train"
            / f"{output_stem}.txt"
        )


        # Save original JPEG bytes.
        with open(output_image, "wb") as f:
            f.write(image_bytes)


        yolo_lines = []

        for left, top, width, height in boxes:

            # Clip bounding box to image boundaries.
            x1 = max(0.0, left)
            y1 = max(0.0, top)

            x2 = min(
                float(image_width),
                left + width
            )

            y2 = min(
                float(image_height),
                top + height
            )

            clipped_width = x2 - x1
            clipped_height = y2 - y1

            if (
                clipped_width <= 0
                or clipped_height <= 0
            ):
                continue


            # Convert:
            #
            # left, top, width, height
            #
            # to YOLO:
            #
            # x_center, y_center,
            # width, height
            #
            # normalized to 0~1.

            x_center = (
                x1 + clipped_width / 2
            ) / image_width

            y_center = (
                y1 + clipped_height / 2
            ) / image_height

            norm_width = (
                clipped_width / image_width
            )

            norm_height = (
                clipped_height / image_height
            )


            yolo_lines.append(

                "0 "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{norm_width:.6f} "
                f"{norm_height:.6f}"
            )


        if not yolo_lines:

            # Normally should not happen because
            # only frames containing car were selected.
            output_image.unlink(
                missing_ok=True
            )

            continue


        output_label.write_text(
            "\n".join(yolo_lines) + "\n"
        )

        detrac_car_instances += len(
            yolo_lines
        )


        if (
            index % 50 == 0
            or index == len(selected)
        ):
            print(
                f"Processed "
                f"{index}/{len(selected)}"
            )


# ============================================================
# Add original custom dataset
# ============================================================

print()
print("Copying original custom dataset...")

custom_train_count = copy_custom_split(
    "train"
)

custom_val_count = copy_custom_split(
    "val"
)


# ============================================================
# Create Experiment 3 YAML
# ============================================================

yaml_content = f"""path: {OUTPUT_DATASET}

train: images/train
val: images/val

nc: 1

names:
  0: car
"""

OUTPUT_YAML.write_text(
    yaml_content
)


# ============================================================
# Final statistics
# ============================================================

final_train_images = len(
    list(
        (
            OUTPUT_DATASET
            / "images"
            / "train"
        ).iterdir()
    )
)

final_train_labels = len(
    list(
        (
            OUTPUT_DATASET
            / "labels"
            / "train"
        ).glob("*.txt")
    )
)

final_val_images = len(
    list(
        (
            OUTPUT_DATASET
            / "images"
            / "val"
        ).iterdir()
    )
)

final_val_labels = len(
    list(
        (
            OUTPUT_DATASET
            / "labels"
            / "val"
        ).glob("*.txt")
    )
)


print()
print("====================================")
print("Experiment 3 Dataset Created")
print("====================================")

print(
    f"UA-DETRAC images selected: "
    f"{len(selected)}"
)

print(
    f"UA-DETRAC car instances: "
    f"{detrac_car_instances}"
)

print(
    f"Original custom train images: "
    f"{custom_train_count}"
)

print(
    f"Original validation images: "
    f"{custom_val_count}"
)

print()

print(
    f"Final train images: "
    f"{final_train_images}"
)

print(
    f"Final train labels: "
    f"{final_train_labels}"
)

print(
    f"Final validation images: "
    f"{final_val_images}"
)

print(
    f"Final validation labels: "
    f"{final_val_labels}"
)

print()

print(
    f"Dataset location:\n"
    f"{OUTPUT_DATASET}"
)

print()

print(
    f"YAML location:\n"
    f"{OUTPUT_YAML}"
)