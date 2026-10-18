import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


# ============================================================
# SESSION SETTINGS
# ============================================================

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/demo_quality_dataset")

SEED = 2026

# You only have 5 images per class for the demo.
TARGET_PER_CLASS = 5

# If you want more demo images from the same 5 source images,
# change this to 2 or 3.
# Example:
# 5 source images + VARIANTS_PER_SOURCE = 2 => 10 images per class
VARIANTS_PER_SOURCE = 1

SAVE_PREPROCESSED = True

TARGET_SIZE = (640, 640)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

DEFECT_TYPES = [
    "blur",
    "glare",
    "skew",
    "occlusion",
    "exposure",
]

GOOD_AUGMENTATIONS = [
    "none",
    "none",
    "slight_rotation",
    "slight_rotation",
    "upside_down",
    "rotate_90",
    "rotate_270",
]


# ============================================================
# Pillow resampling compatibility
# ============================================================

try:
    BILINEAR = Image.Resampling.BILINEAR
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BILINEAR = Image.BILINEAR
    BICUBIC = Image.BICUBIC


# ============================================================
# Helper functions
# ============================================================

def safe_name(text):
    """
    Convert class folder name into a safe filename part.
    Example:
        "passport children" -> "passport_children"
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9а-яА-ЯёЁ]+", "_", text)
    return text


def detect_classes(input_dir):
    """
    Detect class folders automatically.

    If input_dir has subfolders, each subfolder is treated as a class.
    Example:
        data/raw/inn
        data/raw/snils
        data/raw/passport

    If input_dir has images directly, one class called "document" is used.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    images_by_class = {}

    if subdirs:
        for folder in subdirs:
            paths = [
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if paths:
                images_by_class[folder.name] = sorted(paths)

    else:
        paths = [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if paths:
            images_by_class["document"] = sorted(paths)

    if not images_by_class:
        raise FileNotFoundError(
            f"No images found in {input_dir}. "
            f"Please add images into class folders, for example: "
            f"data/raw/inn, data/raw/snils, data/raw/passport."
        )

    return images_by_class


def preprocess_image(path):
    """
    Preprocess image:
    - open image
    - convert to RGB
    - resize to 640x640
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(TARGET_SIZE, BILINEAR)
    return img


# ============================================================
# Good image augmentation
# ============================================================

def augment_good(img, rng):
    """
    Good augmentation.
    The image must remain readable.
    """
    augmentation = rng.choice(GOOD_AUGMENTATIONS)

    if augmentation == "none":
        return img, "none"

    if augmentation == "slight_rotation":
        angle = rng.choice([-1, 1]) * rng.uniform(3, 8)
        img = img.rotate(
            angle,
            resample=BICUBIC,
            expand=False,
            fillcolor=(255, 255, 255),
        )
        return img, f"slight_rotation_{angle:.1f}"

    if augmentation == "upside_down":
        img = img.rotate(180)
        return img, "upside_down"

    if augmentation == "rotate_90":
        img = img.rotate(90, expand=True)
        return img, "rotate_90"

    if augmentation == "rotate_270":
        img = img.rotate(270, expand=True)
        return img, "rotate_270"

    return img, "none"


# ============================================================
# Bad image augmentation / defects
# ============================================================

def apply_blur(img, rng):
    """
    Bad defect: camera blur.
    """
    radius = rng.uniform(3.0, 6.0)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_glare(img, rng):
    """
    Bad defect: glare / reflection.
    """
    width, height = img.size

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(1, 3)):
        x0 = rng.randint(0, int(width * 0.7))
        y0 = rng.randint(0, int(height * 0.7))

        x1 = min(width, x0 + rng.randint(int(width * 0.2), int(width * 0.5)))
        y1 = min(height, y0 + rng.randint(int(height * 0.2), int(height * 0.5)))

        alpha = rng.randint(150, 230)

        draw.ellipse(
            [x0, y0, x1, y1],
            fill=(255, 255, 255, alpha),
        )

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def apply_skew(img, rng):
    """
    Bad defect: strong skew / rotation.
    """
    angle = rng.choice([-1, 1]) * rng.uniform(15, 35)

    return img.rotate(
        angle,
        resample=BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def apply_occlusion(img, rng):
    """
    Bad defect: finger or object covering part of the document.
    """
    width, height = img.size

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(1, 2)):
        x0 = rng.randint(0, int(width * 0.7))
        y0 = rng.randint(0, int(height * 0.7))

        x1 = min(width, x0 + rng.randint(int(width * 0.15), int(width * 0.45)))
        y1 = min(height, y0 + rng.randint(int(height * 0.08), int(height * 0.25)))

        alpha = rng.randint(180, 235)

        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(20, 20, 20, alpha),
        )

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def apply_exposure(img, rng):
    """
    Bad defect: bad exposure.
    Too dark, too bright, or low contrast.
    """
    mode = rng.choice(["dark", "bright", "low_contrast"])

    if mode == "dark":
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.35, 0.55))

    elif mode == "bright":
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(1.6, 2.2))

    else:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.35, 0.6))

    return img


DEFECT_FUNCTIONS = {
    "blur": apply_blur,
    "glare": apply_glare,
    "skew": apply_skew,
    "occlusion": apply_occlusion,
    "exposure": apply_exposure,
}


# ============================================================
# Main demo dataset builder
# ============================================================

def build_demo_dataset():
    rng = random.Random(SEED)

    output_dir = Path(OUTPUT_DIR)

    good_dir = output_dir / "good"
    bad_dir = output_dir / "bad"

    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_PREPROCESSED:
        preprocessed_root = output_dir / "preprocessed"
        preprocessed_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Detect classes and images
    # --------------------------------------------------------
    images_by_class = detect_classes(INPUT_DIR)

    input_counts = {
        class_name: len(paths)
        for class_name, paths in images_by_class.items()
    }

    print("Detected classes:")
    for class_name, count in input_counts.items():
        print(f"  {class_name}: {count} images")
    print()

    min_available = min(input_counts.values())

    if VARIANTS_PER_SOURCE < 1:
        variants = 1
    else:
        variants = VARIANTS_PER_SOURCE

    max_possible_per_class = min_available * variants

    target_per_class = TARGET_PER_CLASS

    if target_per_class > max_possible_per_class:
        print(
            f"Warning: TARGET_PER_CLASS={target_per_class} is too large "
            f"for the available images. "
            f"Maximum possible per class = {max_possible_per_class}. "
            f"Using {max_possible_per_class} instead."
        )
        target_per_class = max_possible_per_class

    if target_per_class < 1:
        raise ValueError("Not enough images to create the dataset.")

    records = []
    bad_index = 0

    metadata_path = output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as metadata_file:

        for class_name, paths in images_by_class.items():
            safe_class = safe_name(class_name)

            paths = paths.copy()
            rng.shuffle(paths)

            needed_sources = math.ceil(target_per_class / variants)

            if needed_sources > len(paths):
                needed_sources = len(paths)

            selected_paths = paths[:needed_sources]

            if SAVE_PREPROCESSED:
                class_preprocessed_dir = (
                    output_dir / "preprocessed" / safe_class
                )
                class_preprocessed_dir.mkdir(parents=True, exist_ok=True)

            class_index = 0

            for source_path in selected_paths:
                if class_index >= target_per_class:
                    break

                # Split is only stored in metadata.
                # Final folders remain good/ and bad/.
                split = rng.choices(
                    ["train", "validation", "test"],
                    weights=[0.7, 0.15, 0.15],
                )[0]

                # --------------------------------------------
                # Preprocess image
                # --------------------------------------------
                img = preprocess_image(source_path)

                if SAVE_PREPROCESSED:
                    preprocessed_path = (
                        output_dir
                        / "preprocessed"
                        / safe_class
                        / f"preprocessed_{safe_class}_{source_path.stem}.jpg"
                    )
                    img.save(preprocessed_path, quality=95)

                source_id = source_path.stem

                # --------------------------------------------
                # Create variants
                # --------------------------------------------
                for variant in range(1, variants + 1):
                    if class_index >= target_per_class:
                        break

                    class_index += 1

                    # ----------------------------------------
                    # GOOD augmented image
                    # ----------------------------------------
                    good_img, good_augmentation = augment_good(img.copy(), rng)

                    if good_img.size != TARGET_SIZE:
                        good_img = good_img.resize(TARGET_SIZE, BILINEAR)

                    good_name = (
                        f"good_{safe_class}_"
                        f"{class_index:03d}_v{variant}.jpg"
                    )
                    good_path = good_dir / good_name

                    good_img.save(good_path, quality=95)

                    good_record = {
                        "image_id": Path(good_name).stem,
                        "class_name": class_name,
                        "safe_class": safe_class,
                        "quality_label": "good",
                        "defect_type": "none",
                        "good_augmentation": good_augmentation,
                        "variant": variant,
                        "source_id": source_id,
                        "source_path": str(source_path),
                        "split": split,
                        "seed": SEED,
                        "image_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
                        "image_path": f"good/{good_name}",
                    }

                    metadata_file.write(
                        json.dumps(good_record, ensure_ascii=False) + "\n"
                    )
                    records.append(good_record)

                    # ----------------------------------------
                    # BAD augmented image
                    # ----------------------------------------
                    defect_type = DEFECT_TYPES[
                        bad_index % len(DEFECT_TYPES)
                    ]
                    bad_index += 1

                    bad_img = DEFECT_FUNCTIONS[defect_type](img.copy(), rng)

                    bad_name = (
                        f"bad_{safe_class}_{defect_type}_"
                        f"{class_index:03d}_v{variant}.jpg"
                    )
                    bad_path = bad_dir / bad_name

                    bad_img.save(bad_path, quality=95)

                    bad_record = {
                        "image_id": Path(bad_name).stem,
                        "class_name": class_name,
                        "safe_class": safe_class,
                        "quality_label": "bad",
                        "defect_type": defect_type,
                        "good_augmentation": "none",
                        "variant": variant,
                        "source_id": source_id,
                        "source_path": str(source_path),
                        "split": split,
                        "seed": SEED,
                        "image_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
                        "image_path": f"bad/{bad_name}",
                    }

                    metadata_file.write(
                        json.dumps(bad_record, ensure_ascii=False) + "\n"
                    )
                    records.append(bad_record)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary = {
        "input_counts": input_counts,
        "target_per_class": target_per_class,
        "variants_per_source": variants,
        "total_records": len(records),
        "quality_label": dict(Counter(r["quality_label"] for r in records)),
        "class_all": dict(Counter(r["class_name"] for r in records)),
        "class_good": dict(
            Counter(
                r["class_name"]
                for r in records
                if r["quality_label"] == "good"
            )
        ),
        "class_bad": dict(
            Counter(
                r["class_name"]
                for r in records
                if r["quality_label"] == "bad"
            )
        ),
        "defect_type_bad": dict(
            Counter(
                r["defect_type"]
                for r in records
                if r["quality_label"] == "bad"
            )
        ),
        # slight rotations are grouped, the exact angle stays in metadata
        "good_augmentation": dict(
            Counter(
                re.sub(r"^slight_rotation_.*", "slight_rotation", r["good_augmentation"])
                for r in records
                if r["quality_label"] == "good"
            )
        ),
        "split": dict(Counter(r["split"] for r in records)),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset created: {output_dir}")
    print(f"Good folder: {good_dir}")
    print(f"Bad folder: {bad_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")
    print()
    print("Quality balance:", summary["quality_label"])
    print("Class balance good:", summary["class_good"])
    print("Class balance bad:", summary["class_bad"])
    print("Bad defect balance:", summary["defect_type_bad"])
    print("Good augmentation balance:", summary["good_augmentation"])


# ============================================================
# Run
# ============================================================

def parse_args():
    """
    Command line overrides for the session settings above.
    Demo:     python build_good_bad_dataset.py
    Homework: python build_good_bad_dataset.py --output data/quality_dataset_300 --target 300 --variants 3
    """
    parser = argparse.ArgumentParser(description="Build a balanced good/bad quality dataset.")
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--variants", type=int, default=VARIANTS_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    TARGET_PER_CLASS = args.target
    VARIANTS_PER_SOURCE = args.variants
    SEED = args.seed
    build_demo_dataset()