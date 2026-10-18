import argparse
import json
from pathlib import Path

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    print("Please install PyTorch and torchvision:")
    print("pip install torch torchvision")
    raise


CLASS_TO_LABEL = {
    "good": 0,
    "bad": 1,
}

LABEL_TO_CLASS = {
    0: "good",
    1: "bad",
}


def default_transform():
    """
    Default transform for CNN training.

    Images are already 640x640, but we resize to 224x224
    for faster training with common pretrained models.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class QualityDataset(Dataset):
    """
    PyTorch Dataset for good/bad document quality classification.
    """

    def __init__(
        self,
        dataset_dir,
        split=None,
        transform=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = self.dataset_dir / "metadata.jsonl"

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found in {self.dataset_dir}. "
                f"Please generate the dataset first."
            )

        self.transform = transform or default_transform()
        self.records = []

        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line in metadata_file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                # Filter by split if needed
                if split is not None and record.get("split") != split:
                    continue

                # Validate label
                quality_label = record.get("quality_label")
                if quality_label not in CLASS_TO_LABEL:
                    continue

                # Validate image path
                image_path = self.dataset_dir / record.get("image_path", "")
                if not image_path.exists():
                    continue

                record["_full_image_path"] = image_path
                self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = record["_full_image_path"]
        image = Image.open(image_path).convert("RGB")

        label = CLASS_TO_LABEL[record["quality_label"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def show_batch(loader, title):
    """Load one batch and print its shape and labels."""
    images, labels = next(iter(loader))

    print(f"[{title}] batch loaded successfully.")
    print(f"  Images tensor shape: {images.shape}")
    print(f"  Labels tensor shape: {labels.shape}")
    print(f"  Labels: {labels.tolist()}")
    print(f"  Label names: {[LABEL_TO_CLASS[label] for label in labels.tolist()]}")


def main():
    parser = argparse.ArgumentParser(
        description="Test PyTorch data loading for quality dataset."
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/demo_quality_dataset"),
        help="Folder containing good/, bad/, and metadata.jsonl",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["all", "train", "validation", "test"],
        help=(
            "Load one split, or 'all' to load every record as one dataset. "
            "If not set, train, validation and test are loaded separately."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed for DataLoader shuffling",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Batch size: {args.batch_size}")
    print()

    if args.split is not None:
        split = None if args.split == "all" else args.split
        splits = {args.split: QualityDataset(args.dataset_dir, split=split)}
    else:
        splits = {
            name: QualityDataset(args.dataset_dir, split=name)
            for name in ("train", "validation", "test")
        }

    print("Number of samples:")
    for name, dataset in splits.items():
        labels = [record["quality_label"] for record in dataset.records]
        print(
            f"  {name:<10} {len(dataset):>5}  "
            f"(good={labels.count('good')}, bad={labels.count('bad')})"
        )
    print(f"  {'total':<10} {sum(len(d) for d in splits.values()):>5}")
    print()

    for name, dataset in splits.items():
        if len(dataset) == 0:
            raise ValueError(
                f"No records found in {args.dataset_dir} for split={name}. "
                f"For a small demo dataset use --split all."
            )

        # shuffle only the training data; validation/test keep a fixed order
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name in ("train", "all")),
            num_workers=0,
        )
        show_batch(loader, name)


if __name__ == "__main__":
    main()