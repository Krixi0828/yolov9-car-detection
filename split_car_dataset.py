from pathlib import Path
import random
import shutil

random.seed(42)

source = Path("data/car_raw")
target = Path("data/car_dataset")

image_dir = source / "images"
label_dir = source / "labels"

valid_extensions = {".jpg", ".jpeg", ".webp"}

images = [
    p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in valid_extensions
]

random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]


def copy_pairs(image_list, split):
    for image_path in image_list:

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Missing label: {image_path.name}")
            continue

        shutil.copy2(
            image_path,
            target / "images" / split / image_path.name
        )

        shutil.copy2(
            label_path,
            target / "labels" / split / label_path.name
        )


copy_pairs(train_images, "train")
copy_pairs(val_images, "val")

print(f"Total: {len(images)}")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")