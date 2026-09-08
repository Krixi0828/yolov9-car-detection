from pathlib import Path
from collections import defaultdict
import csv
import re

from PIL import Image


# ============================================================
# Settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_IMAGES = PROJECT_ROOT / "data" / "car_test" / "images"
GT_LABELS = PROJECT_ROOT / "data" / "car_test" / "labels"

PREDICTION_ROOT = (
    PROJECT_ROOT
    / "yolov9"
    / "runs"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "test_error_analysis_exp4_640"
)

# Diagnostic operating point.
#
# This comes from the Test F1 curve:
# F1 ≈ 0.64 at confidence ≈ 0.217
#
# This is ONLY for error analysis.
# It does not replace the official mAP evaluation.
CONF_THRESHOLD = 0.217

# A detection is considered a correct match if:
# IoU >= 0.5
IOU_THRESHOLD = 0.5

# Training / inference input size
INPUT_SIZE = 320


# ============================================================
# Find prediction directory
# ============================================================

prediction_runs = [
    p
    for p in PREDICTION_ROOT.glob("car_exp4_640_error_analysis*")
    if (p / "labels").exists()
]

if not prediction_runs:
    raise FileNotFoundError(
        "Cannot find prediction labels.\n"
        "Run val_dual.py with --save-txt --save-conf first."
    )

prediction_run = max(
    prediction_runs,
    key=lambda p: p.stat().st_mtime
)

PRED_LABELS = prediction_run / "labels"

print("Using predictions from:")
print(prediction_run)
print()


# ============================================================
# Helpers
# ============================================================

def yolo_to_xyxy(xc, yc, w, h):
    """
    Convert normalized YOLO:

        x_center, y_center, width, height

    into normalized:

        x1, y1, x2, y2
    """

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return (x1, y1, x2, y2)


def box_iou(box1, box2):
    """
    IoU between two normalized xyxy boxes.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * \
            max(0.0, box1[3] - box1[1])

    area2 = max(0.0, box2[2] - box2[0]) * \
            max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(label_path):
    """
    GT format:

    class xc yc w h
    """

    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        # We only evaluate class 0 = car
        if class_id != 0:
            continue

        xc, yc, w, h = map(float, parts[1:])

        boxes.append({
            "box": yolo_to_xyxy(xc, yc, w, h),
            "w": w,
            "h": h,
        })

    return boxes


def load_predictions(label_path):
    """
    Prediction format with --save-conf:

    class xc yc w h confidence
    """

    predictions = []

    if not label_path.exists():
        return predictions

    for line in label_path.read_text().splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 6:
            continue

        class_id = int(parts[0])

        if class_id != 0:
            continue

        xc, yc, w, h, conf = map(
            float,
            parts[1:6]
        )

        if conf < CONF_THRESHOLD:
            continue

        predictions.append({
            "box": yolo_to_xyxy(xc, yc, w, h),
            "confidence": conf,
        })

    # Highest-confidence predictions first
    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return predictions


def get_sequence(image_stem):
    """
    Example:

    detrac_test_MVI_39031_img00001
                     ↓
                  MVI_39031
    """

    match = re.search(
        r"detrac_test_(MVI_\d+)_img\d+",
        image_stem
    )

    if match:
        return match.group(1)

    return "unknown"


def get_size_category(gt, image_width, image_height):
    """
    Estimate object size AFTER resize/letterbox scaling
    to the 320x320 model input.

    COCO-like area thresholds are used:

        small  < 32^2 pixels
        medium < 96^2 pixels
        large  >= 96^2 pixels

    This is an error-analysis definition,
    not an official COCO evaluation.
    """

    gt_width_px = gt["w"] * image_width
    gt_height_px = gt["h"] * image_height

    scale = min(
        INPUT_SIZE / image_width,
        INPUT_SIZE / image_height
    )

    scaled_width = gt_width_px * scale
    scaled_height = gt_height_px * scale

    area = scaled_width * scaled_height

    if area < 32 ** 2:
        return "small"

    if area < 96 ** 2:
        return "medium"

    return "large"


# ============================================================
# Matching
# ============================================================

def match_predictions(gt_boxes, predictions):
    """
    Greedy one-to-one matching.

    Each prediction can match at most one GT.
    Each GT can match at most one prediction.

    Match condition:
        IoU >= 0.5
    """

    matched_gt = set()

    tp = 0
    fp = 0

    for pred in predictions:

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(gt_boxes):

            if gt_index in matched_gt:
                continue

            iou = box_iou(
                pred["box"],
                gt["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= IOU_THRESHOLD
        ):
            matched_gt.add(best_gt_index)
            tp += 1

        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn, matched_gt


# ============================================================
# Prepare output
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


size_stats = {
    "small": {
        "gt": 0,
        "tp": 0,
        "fn": 0,
    },
    "medium": {
        "gt": 0,
        "tp": 0,
        "fn": 0,
    },
    "large": {
        "gt": 0,
        "tp": 0,
        "fn": 0,
    },
}


sequence_stats = defaultdict(
    lambda: {
        "images": 0,
        "gt": 0,
        "predictions": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
)


overall = {
    "images": 0,
    "gt": 0,
    "predictions": 0,
    "tp": 0,
    "fp": 0,
    "fn": 0,
}


# ============================================================
# Analyze every Test image
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

images = sorted([
    p
    for p in TEST_IMAGES.iterdir()
    if p.is_file()
    and p.suffix.lower() in image_extensions
])


for image_path in images:

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    gt_path = GT_LABELS / f"{image_path.stem}.txt"

    pred_path = (
        PRED_LABELS
        / f"{image_path.stem}.txt"
    )

    gt_boxes = load_ground_truth(gt_path)

    predictions = load_predictions(
        pred_path
    )

    tp, fp, fn, matched_gt = (
        match_predictions(
            gt_boxes,
            predictions
        )
    )


    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    overall["images"] += 1
    overall["gt"] += len(gt_boxes)
    overall["predictions"] += len(predictions)

    overall["tp"] += tp
    overall["fp"] += fp
    overall["fn"] += fn


    # --------------------------------------------------------
    # Object-size statistics
    # --------------------------------------------------------

    for gt_index, gt in enumerate(gt_boxes):

        category = get_size_category(
            gt,
            image_width,
            image_height
        )

        size_stats[category]["gt"] += 1

        if gt_index in matched_gt:
            size_stats[category]["tp"] += 1
        else:
            size_stats[category]["fn"] += 1


    # --------------------------------------------------------
    # Sequence statistics
    # --------------------------------------------------------

    sequence = get_sequence(
        image_path.stem
    )

    stat = sequence_stats[sequence]

    stat["images"] += 1
    stat["gt"] += len(gt_boxes)
    stat["predictions"] += len(predictions)

    stat["tp"] += tp
    stat["fp"] += fp
    stat["fn"] += fn


# ============================================================
# Overall result
# ============================================================

precision = (
    overall["tp"]
    / (overall["tp"] + overall["fp"])
    if overall["tp"] + overall["fp"] > 0
    else 0
)

recall = (
    overall["tp"]
    / (overall["tp"] + overall["fn"])
    if overall["tp"] + overall["fn"] > 0
    else 0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if precision + recall > 0
    else 0
)


print("====================================")
print("Overall Diagnostic Result")
print("====================================")

print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IoU threshold: {IOU_THRESHOLD}")

print()

print(f"Images: {overall['images']}")
print(f"GT cars: {overall['gt']}")
print(f"Predictions: {overall['predictions']}")

print()

print(f"TP: {overall['tp']}")
print(f"FP: {overall['fp']}")
print(f"FN: {overall['fn']}")

print()

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")


# ============================================================
# Object-size analysis
# ============================================================

print()
print("====================================")
print("Object-size Analysis")
print("====================================")

size_csv = OUTPUT_DIR / "object_size_analysis.csv"

with open(
    size_csv,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "size",
        "gt_count",
        "tp",
        "fn",
        "recall",
        "fn_rate",
    ])

    for category in [
        "small",
        "medium",
        "large",
    ]:

        stat = size_stats[category]

        recall_size = (
            stat["tp"] / stat["gt"]
            if stat["gt"] > 0
            else 0
        )

        fn_rate = (
            stat["fn"] / stat["gt"]
            if stat["gt"] > 0
            else 0
        )

        print(
            f"{category:6s} | "
            f"GT={stat['gt']:4d} | "
            f"TP={stat['tp']:4d} | "
            f"FN={stat['fn']:4d} | "
            f"Recall={recall_size:.3f}"
        )

        writer.writerow([
            category,
            stat["gt"],
            stat["tp"],
            stat["fn"],
            f"{recall_size:.6f}",
            f"{fn_rate:.6f}",
        ])


# ============================================================
# Sequence-level analysis
# ============================================================

rows = []

for sequence, stat in sequence_stats.items():

    seq_precision = (
        stat["tp"]
        / (stat["tp"] + stat["fp"])
        if stat["tp"] + stat["fp"] > 0
        else 0
    )

    seq_recall = (
        stat["tp"]
        / (stat["tp"] + stat["fn"])
        if stat["tp"] + stat["fn"] > 0
        else 0
    )

    seq_f1 = (
        2 * seq_precision * seq_recall
        / (seq_precision + seq_recall)
        if seq_precision + seq_recall > 0
        else 0
    )

    rows.append({
        "sequence": sequence,
        "images": stat["images"],
        "gt": stat["gt"],
        "predictions": stat["predictions"],
        "tp": stat["tp"],
        "fp": stat["fp"],
        "fn": stat["fn"],
        "precision": seq_precision,
        "recall": seq_recall,
        "f1": seq_f1,
    })


rows.sort(
    key=lambda x: (
        x["recall"],
        x["f1"]
    )
)


sequence_csv = (
    OUTPUT_DIR / "sequence_analysis.csv"
)

with open(
    sequence_csv,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sequence",
            "images",
            "gt",
            "predictions",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]
    )

    writer.writeheader()

    for row in rows:

        output_row = row.copy()

        output_row["precision"] = (
            f"{row['precision']:.6f}"
        )

        output_row["recall"] = (
            f"{row['recall']:.6f}"
        )

        output_row["f1"] = (
            f"{row['f1']:.6f}"
        )

        writer.writerow(output_row)


print()
print("====================================")
print("10 Hardest Sequences by Recall")
print("====================================")

for row in rows[:10]:

    print(
        f"{row['sequence']:10s} | "
        f"GT={row['gt']:3d} | "
        f"TP={row['tp']:3d} | "
        f"FP={row['fp']:3d} | "
        f"FN={row['fn']:3d} | "
        f"P={row['precision']:.3f} | "
        f"R={row['recall']:.3f} | "
        f"F1={row['f1']:.3f}"
    )


print()
print("Saved:")
print(size_csv)
print(sequence_csv)