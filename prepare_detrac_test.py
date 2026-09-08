from pathlib import Path
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

DETRAC_TEST_XML_DIR = (
    DETRAC_RAW / "DETRAC-Test-Annotations-XML"
)

OUTPUT_TEST = DATA_DIR / "car_test"

OUTPUT_YAML = DATA_DIR / "car_exp3_eval.yaml"


# ============================================================
# Test settings
# ============================================================

IMAGES_PER_SEQUENCE = 5

POSITIVE_PER_SEQUENCE = 4
NEGATIVE_PER_SEQUENCE = 1


# ============================================================
# Helper
# ============================================================

def evenly_sample(items, n):
    """
    從一個 sequence 中平均抽取 n 個 frame，
    避免抽到大量相鄰 frames。
    """

    if n <= 0:
        return []

    if n >= len(items):
        return items.copy()

    if n == 1:
        return [items[len(items) // 2]]

    indices = [
        round(i * (len(items) - 1) / (n - 1))
        for i in range(n)
    ]

    return [items[i] for i in indices]


# ============================================================
# Prepare output directory
# ============================================================

if OUTPUT_TEST.exists():
    print(f"Removing old test set: {OUTPUT_TEST}")
    shutil.rmtree(OUTPUT_TEST)

IMAGE_OUTPUT = OUTPUT_TEST / "images"
LABEL_OUTPUT = OUTPUT_TEST / "labels"

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
LABEL_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read Test XML
# ============================================================

xml_files = sorted(
    DETRAC_TEST_XML_DIR.glob("MVI_*.xml")
)

if not xml_files:
    raise FileNotFoundError(
        f"No Test XML files found in "
        f"{DETRAC_TEST_XML_DIR}"
    )

print("Test sequences:", len(xml_files))


selected = []

total_positive_candidates = 0
total_negative_candidates = 0


for xml_path in xml_files:

    sequence = xml_path.stem

    root = ET.parse(xml_path).getroot()

    positive_frames = []
    negative_frames = []

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
                "vehicle_type",
                "unknown"
            )

            # Keep ONLY car.
            #
            # van / bus / others are not car.
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

        frame_info = {
            "sequence": sequence,
            "frame_num": frame_num,
            "boxes": car_boxes,
        }

        if car_boxes:
            positive_frames.append(frame_info)
        else:
            negative_frames.append(frame_info)

    total_positive_candidates += len(
        positive_frames
    )

    total_negative_candidates += len(
        negative_frames
    )


    # --------------------------------------------------------
    # Prefer:
    #
    # 4 positive + 1 negative
    # --------------------------------------------------------

    chosen_positive = evenly_sample(
        positive_frames,
        min(
            POSITIVE_PER_SEQUENCE,
            len(positive_frames)
        )
    )

    chosen_negative = evenly_sample(
        negative_frames,
        min(
            NEGATIVE_PER_SEQUENCE,
            len(negative_frames)
        )
    )


    chosen = (
        chosen_positive
        + chosen_negative
    )


    # --------------------------------------------------------
    # If the sequence has no negative frame,
    # use more positive frames to reach 5.
    # --------------------------------------------------------

    if len(chosen) < IMAGES_PER_SEQUENCE:

        already_selected = {
            item["frame_num"]
            for item in chosen
        }

        remaining = [
            item
            for item in positive_frames + negative_frames
            if item["frame_num"]
            not in already_selected
        ]

        need = (
            IMAGES_PER_SEQUENCE
            - len(chosen)
        )

        chosen += evenly_sample(
            remaining,
            min(need, len(remaining))
        )


    selected.extend(chosen)


print(
    "Positive candidate frames:",
    total_positive_candidates
)

print(
    "Negative candidate frames:",
    total_negative_candidates
)

print(
    "Selected test images:",
    len(selected)
)


# ============================================================
# Extract ONLY selected test images
# ============================================================

positive_count = 0
negative_count = 0
car_instance_count = 0


with zipfile.ZipFile(DETRAC_ZIP, "r") as zf:

    members = set(zf.namelist())

    for index, item in enumerate(
        selected,
        start=1
    ):

        sequence = item["sequence"]
        frame_num = item["frame_num"]
        boxes = item["boxes"]

        image_name = (
            f"img{frame_num:05d}.jpg"
        )

        zip_member = (
            f"DETRAC-Images/"
            f"{sequence}/"
            f"{image_name}"
        )

        if zip_member not in members:
            raise FileNotFoundError(
                f"Image not found in ZIP: "
                f"{zip_member}"
            )

        image_bytes = zf.read(zip_member)

        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            image_width, image_height = (
                image.size
            )


        output_stem = (
            f"detrac_test_"
            f"{sequence}_"
            f"img{frame_num:05d}"
        )

        output_image = (
            IMAGE_OUTPUT
            / f"{output_stem}.jpg"
        )

        output_label = (
            LABEL_OUTPUT
            / f"{output_stem}.txt"
        )


        # Save image
        with open(output_image, "wb") as f:
            f.write(image_bytes)


        # ----------------------------------------------------
        # Positive image
        # ----------------------------------------------------

        if boxes:

            positive_count += 1

            yolo_lines = []

            for left, top, width, height in boxes:

                # Clip box to image boundary
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


                # YOLO normalization
                x_center = (
                    x1 + clipped_width / 2
                ) / image_width

                y_center = (
                    y1 + clipped_height / 2
                ) / image_height

                norm_width = (
                    clipped_width
                    / image_width
                )

                norm_height = (
                    clipped_height
                    / image_height
                )


                yolo_lines.append(
                    "0 "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{norm_width:.6f} "
                    f"{norm_height:.6f}"
                )


            output_label.write_text(
                "\n".join(yolo_lines)
                + ("\n" if yolo_lines else "")
            )

            car_instance_count += len(
                yolo_lines
            )


        # ----------------------------------------------------
        # Negative image
        #
        # Empty .txt is a valid YOLO label:
        # this image contains no target class "car".
        # ----------------------------------------------------

        else:

            negative_count += 1

            output_label.write_text("")


        if (
            index % 25 == 0
            or index == len(selected)
        ):

            print(
                f"Processed "
                f"{index}/{len(selected)}"
            )


# ============================================================
# Create evaluation YAML
# ============================================================

EXP3_DATASET = (
    DATA_DIR / "car_dataset_exp3"
)

yaml_content = f"""path: {DATA_DIR}

train: car_dataset_exp3/images/train
val: car_dataset_exp3/images/val
test: car_test/images

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

image_count = len(
    list(IMAGE_OUTPUT.glob("*.jpg"))
)

label_count = len(
    list(LABEL_OUTPUT.glob("*.txt"))
)


print()
print("====================================")
print("UA-DETRAC Test Set Created")
print("====================================")

print(
    f"Test sequences: "
    f"{len(xml_files)}"
)

print(
    f"Test images: "
    f"{image_count}"
)

print(
    f"Test labels: "
    f"{label_count}"
)

print(
    f"Positive images: "
    f"{positive_count}"
)

print(
    f"Negative images: "
    f"{negative_count}"
)

print(
    f"Car instances: "
    f"{car_instance_count}"
)

print()

print(
    f"Test dataset:\n"
    f"{OUTPUT_TEST}"
)

print()

print(
    f"Evaluation YAML:\n"
    f"{OUTPUT_YAML}"
)