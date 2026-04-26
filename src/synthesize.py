import os
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR      = (BASE_DIR / "../dataset").resolve()
ANNOTATIONS_CSV = (BASE_DIR / "../dataset/annotations.csv").resolve()

SYNTHETIC_DIR = (BASE_DIR / "../dataset/synthetic").resolve()
N_SYNTHETIC_PER_CLASS = 50

# Resize every crop to this before pasting (keeps signs a consistent size)
# Tried out different values(started with 80 --> Really small)
SIGN_SIZE = 150       # pixels - will be pasted on a 224*224 canvas

# Canvas size (matches IMAGE_SIZE in dataset.py)
CANVAS_SIZE = 224

# Probability of each augmentation being applied to a single synthesis step
PROB_NOISE       = 0.5
PROB_BRIGHTNESS  = 0.8
PROB_BLUR        = 0.3


def _random_solid_bg():
    """
    Random solid-colour background(Choosing extreme values will make it look so unrealistic)
    """
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    bg = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), (b, g, r), dtype=np.uint8)  # BGR
    return bg


def _random_noise_bg():
    """Random Gaussian noise background."""
    noise = np.random.randint(30, 200, (CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (15, 15), 0)
    return noise


_BG_GENERATORS = [_random_solid_bg, _random_noise_bg]


def _get_random_background():
    return random.choice(_BG_GENERATORS)()


def _apply_brightness_shift(img):
    """
    Simulate different lighting conditions by shifting brightness in HSV space.
    """
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(0.5, 1.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _apply_gaussian_noise(img, std = 15.0):
    """Add Gaussian noise to the sign crop."""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _apply_blur(img):
    """Mild Gaussian blur to simulate motion / focus blur."""
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def _augment_crop(crop):
    """Apply random combination of augmentations to a sign crop."""
    if random.random() < PROB_BRIGHTNESS:
        crop = _apply_brightness_shift(crop)
    if random.random() < PROB_NOISE:
        crop = _apply_gaussian_noise(crop)
    if random.random() < PROB_BLUR:
        crop = _apply_blur(crop)
    return crop


def _paste_on_background(crop, canvas_size = CANVAS_SIZE, sign_size = SIGN_SIZE):
    """
    Resize crop to sign_size * sign_size and paste it at a random position
    on a randomly generated background canvas.
    """
    bg   = _get_random_background()
    sign = cv2.resize(crop, (sign_size, sign_size), interpolation=cv2.INTER_CUBIC)

    # Random position - ensure the sign fits fully inside the canvas
    max_x = canvas_size - sign_size
    max_y = canvas_size - sign_size
    x     = random.randint(0, max_x)
    y     = random.randint(0, max_y)

    bg[y:y + sign_size, x:x + sign_size] = sign
    return bg


def _load_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            entry = {
                "filename": row["filename"].strip(),
                "label":    row["class"].strip(),
            }
            rows.append(entry)
    return rows


def generate_synthetic_images(images_dir=IMAGES_DIR, annotations_csv=ANNOTATIONS_CSV, output_dir=SYNTHETIC_DIR, 
                              n_per_class=N_SYNTHETIC_PER_CLASS, canvas_size=CANVAS_SIZE, sign_size=SIGN_SIZE, 
                              seed=RANDOM_SEED):
    """
    Generate synthetic training images by:
        1. Cropping the real sign region from each source image
        2. Applying random augmentations to the crop
        3. Pasting the augmented crop on a random background

    Returns
    -------
    A list of {'filename': str, 'label': str} dicts pointing to the
    saved synthetic images.
    """
    random.seed(seed)
    np.random.seed(seed)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    # Clean previous synthetic data
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Group source images by class
    rows = _load_csv(annotations_csv)
    class_groups: dict[str, list[dict]] = {}
    for row in rows:
        class_groups.setdefault(row["label"], []).append(row)

    print(f"\n  [synthesize] Generating {n_per_class} images * "
          f"{len(class_groups)} classes = "
          f"{n_per_class * len(class_groups)} total")

    synthetic_rows = []
    counters       = {cls: 0 for cls in class_groups}

    for cls, sources in class_groups.items():
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        generated = 0

        while generated < n_per_class:
            # Pick a random source image for this class
            src = random.choice(sources)
            img_path = images_dir / src["filename"]

            img = cv2.imread(str(img_path))

            if img is None:
                raise FileNotFoundError(f"Missing or corrupted image: {img_path}")

            # Augment crop
            img = _augment_crop(img)

            # Paste on background
            synthetic_img = _paste_on_background(img, canvas_size, sign_size)

            # Save
            fname = f"syn_{cls}_{generated:04d}.png"
            out_path = cls_dir / fname
            cv2.imwrite(str(out_path), synthetic_img)

            synthetic_rows.append({
                "filename": str(Path(output_dir) / cls / fname),
                "label":    cls,
            })
            generated += 1

        counters[cls] = generated
        print(f"    {cls:<20} => {generated} images")

    print(f"\n  [synthesize] Done. Total: {sum(counters.values())} images => {output_dir}")
    return synthetic_rows


def get_synthetic_rows(images_dir=IMAGES_DIR, annotations_csv=ANNOTATIONS_CSV, output_dir=SYNTHETIC_DIR, 
                       n_per_class=N_SYNTHETIC_PER_CLASS, regenerate=False):
    output_path = Path(output_dir)

    already_exists = output_path.exists() and any(output_path.iterdir())
    if already_exists and not regenerate:
        print(f"  [synthesize] Reusing existing synthetic images in '{output_dir}'")
        # Rebuild row list from existing files
        rows = []
        for cls_dir in sorted(output_path.iterdir()):
            if cls_dir.is_dir():
                for img_file in sorted(cls_dir.glob("*.png")):
                    rows.append({
                        "filename": str(img_file),
                        "label":    cls_dir.name,
                    })
        print(f"  [synthesize] Found {len(rows)} existing synthetic images")
        return rows, output_dir

    rows = generate_synthetic_images(
        images_dir=images_dir,
        annotations_csv=annotations_csv,
        output_dir=output_dir,
        n_per_class=n_per_class,
    )
    return rows, output_dir
