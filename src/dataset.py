import csv
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = (BASE_DIR / "../dataset").resolve()
ANNOTATIONS_CSV = (BASE_DIR / "../dataset/annotations.csv").resolve()

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42
IMAGE_SIZE  = 224
BATCH_SIZE  = 32


def _load_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            label    = row["class"].strip()
            if filename and label:
                rows.append({"filename": filename, "label": label})
    if not rows:
        raise ValueError(f"No valid rows found in '{csv_path}'")
    return rows


def _build_label_map(rows):
    # To get the same order of class names(sorted)
    unique_labels = sorted({r["label"] for r in rows})
    return {label: idx for idx, label in enumerate(unique_labels)}


def _split_rows(rows, train_ratio = TRAIN_RATIO, val_ratio = VAL_RATIO, seed = RANDOM_SEED):
    labels = [r["label"] for r in rows]

    train_rows, temp_rows = train_test_split(
        rows,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=labels,
    )

    test_ratio    = 1.0 - train_ratio - val_ratio
    relative_test = test_ratio / (val_ratio + test_ratio)
    temp_labels   = [row["label"] for row in temp_rows]

    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_rows, val_rows, test_rows


class TrafficSignDataset(Dataset):
    """
    Each row has:
        'filename'   - path
        'label'      - class
        'images_dir' - Image Directory.
    """

    def __init__(self,rows,images_dir,label_map,transform=None):
        self.rows       = rows
        self.images_dir = Path(images_dir)
        self.label_map  = label_map
        self.transform  = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row      = self.rows[idx]
        img_path = self.images_dir / row["filename"]
        label    = self.label_map[row["label"]]

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}\n")

        if self.transform:
            img = self.transform(img)

        return img, label

    def class_distribution(self):
        return dict(Counter(r["label"] for r in self.rows))


# Default (no-augmentation) transform
BASELINE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def get_datasets(
    images_dir           = IMAGES_DIR,
    annotations_csv      = ANNOTATIONS_CSV,
    train_transform      = None,
    val_test_transform   = None,
    synthetic_rows       = None,
    synthetic_images_dir = None,
):
    """
    Build train / val / test Dataset objects from the annotations CSV.

    Args:
        synthetic_rows        : list of {'filename', 'label'} dicts from
                                synthesize.get_synthetic_rows().
        synthetic_images_dir  : root directory where synthetic images are stored.

    Returns => train_ds, val_ds, test_ds, label_map, class_names
    """
    if train_transform    is None:
        train_transform    = BASELINE_TRANSFORM
    if val_test_transform is None:
        val_test_transform = BASELINE_TRANSFORM

    rows        = _load_csv(annotations_csv)
    label_map   = _build_label_map(rows)
    # To get the same order of class names(sorted)
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    train_rows, val_rows, test_rows = _split_rows(rows)

    # Add synthetic rows to training dataset
    if synthetic_rows:
        base = synthetic_images_dir or ""
        tagged = []

        for r in synthetic_rows:
            new_row = dict(r)
            new_row["images_dir"] = base
            tagged.append(new_row)

        train_rows = train_rows + tagged
        print(f"  [dataset] Added {len(tagged)} synthetic rows into training set")

    train_ds = TrafficSignDataset(train_rows, images_dir, label_map, train_transform)
    val_ds   = TrafficSignDataset(val_rows,   images_dir, label_map, val_test_transform)
    test_ds  = TrafficSignDataset(test_rows,  images_dir, label_map, val_test_transform)

    return train_ds, val_ds, test_ds, label_map, class_names


def get_dataloaders(train_ds, val_ds, test_ds,batch_size  = BATCH_SIZE):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              pin_memory=True)
    return train_loader, val_loader, test_loader


def print_split_summary(train_ds, val_ds, test_ds, class_names):
    train_len = len(train_ds)
    val_len   = len(val_ds)
    test_len  = len(test_ds)
    total     = train_len + val_len + test_len

    print("\nDATASET SPLIT SUMMARY")
    print("----------------------")
    print(f"Train: {train_len} ({train_len / total * 100:.1f}%)")
    print(f"Val:   {val_len} ({val_len / total * 100:.1f}%)")
    print(f"Test:  {test_len} ({test_len / total * 100:.1f}%)")
    print(f"Total: {total}")

    print("\nCLASS DISTRIBUTION")
    print("------------------")

    train_dist = train_ds.class_distribution()
    val_dist   = val_ds.class_distribution()
    test_dist  = test_ds.class_distribution()

    for cls in class_names:
        print(
            f"{cls}: "
            f"train={train_dist.get(cls, 0)}, "
            f"val={val_dist.get(cls, 0)}, "
            f"test={test_dist.get(cls, 0)}"
        )

    print()