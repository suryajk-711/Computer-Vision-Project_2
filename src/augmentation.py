from torchvision import transforms

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# CONFIGS
ROTATION_DEGREES = 15

COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST   = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE        = 0.05    # small - because large value can change color

# RandomPerspective: distortion scale
PERSPECTIVE_DISTORTION = 0.2
PERSPECTIVE_PROB       = 0.3


def get_baseline_transform():
    """
    Minimal transform: resize => tensor => ImageNet normalise.
    Used for:
        1. Validation and test sets in ALL configs
        2. Training set in Config 1 (baseline, no augmentation)
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_augmented_transform():
    """
    Full augmentation pipeline tuned for traffic-sign images.

    RandomRotation       - camera tilt or non-frontal sign orientation +-15 degree is physically realistic.
    ColorJitter          - handles lighting variation (time of day, weather, camera white-balance differences).
    RandomPerspective    - mild projective distortion to simulate oblique viewing angles.
    ToTensor + Normalize - always last.
    """
    return transforms.Compose([
        # Resize to IMAGE_SIZE*IMAGE_SIZE
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Spatial transforms
        transforms.RandomRotation(degrees=ROTATION_DEGREES),
        transforms.RandomPerspective(
            distortion_scale=PERSPECTIVE_DISTORTION,
            p=PERSPECTIVE_PROB,
        ),

        # Photometric transforms
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=COLOR_JITTER_HUE,
        ),

        # Tensor conversion + normalisation
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform(config):
    """
    Return the appropriate training transform for a given config.

    Config 1 - baseline         : no augmentation
    Config 2 - augmented        : full augmentation
    Config 3 - synthetic        : no augmentation (data diversity via synthesis)
    Config 4 - synth + augmented: full augmentation

    Val / test always use the baseline transform regardless of config.(Only for training)
    """
    if config in (1, 3):
        return get_baseline_transform()
    elif config in (2, 4):
        return get_augmented_transform()
    else:
        raise ValueError(f"Config must be 1-4, got {config}")


def get_val_transform():
    """Val/test transform - always baseline (no augmentation)."""
    return get_baseline_transform()
