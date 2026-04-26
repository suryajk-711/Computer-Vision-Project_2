import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


BACKBONE = "efficientnet_b0"
DROPOUT = 0.3
PRETRAINED = True
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR  = (BASE_DIR / "../results/checkpoints").resolve()


class TrafficSignClassifier(nn.Module):
    """
    ImageNet-pretrained with a custom classification head.

    Architecture
    ------------
    backbone (frozen or unfrozen)
        1. feature extractor (all layers except final classifier)
    head
        1. AdaptiveAvgPool2d  =>  (batch, features, 1, 1)
        2. Flatten            =>  (batch, features)
        3. Dropout(p)
        4. Linear(features, num_classes)
    """

    def __init__(self, num_classes, backbone=BACKBONE, pretrained=PRETRAINED, dropout=DROPOUT):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        self.features    = base.features
        self.pool        = nn.AdaptiveAvgPool2d(1)
        in_features      = base.classifier[1].in_features

        # Custom classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.freeze_backbone()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        """Freeze all backbone parameters. Only head will be trained."""
        for param in self.features.parameters():
            param.requires_grad = False
        print(f"  [model] Backbone frozen - training head only "
              f"({self._count_trainable():,} trainable params)")

    def unfreeze_backbone(self, unfreeze_last_n_blocks = None):
        """
        Unfreeze backbone parameters for fine-tuning.

        Parameters
        ----------
        unfreeze_last_n_blocks : int or None
            If None  => unfreeze everything.
            If int N => unfreeze only the last N children of self.features.
        """
        if unfreeze_last_n_blocks is None:
            for param in self.features.parameters():
                param.requires_grad = True
        else:
            children = list(self.features.children())
            for child in children[:-unfreeze_last_n_blocks]:
                for param in child.parameters():
                    param.requires_grad = False
            for child in children[-unfreeze_last_n_blocks:]:
                for param in child.parameters():
                    param.requires_grad = True

        print(f"  [model] Backbone unfrozen - "
              f"{self._count_trainable():,} trainable params")

    def _count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


def get_model(num_classes, backbone=BACKBONE, pretrained=PRETRAINED, dropout=DROPOUT, device=None):
    """
    Build and return a TrafficSignClassifier.

    Parameters
    ----------
    num_classes : number of output classes
    backbone    : "efficientnet_b0"
    pretrained  : load ImageNet weights
    dropout     : dropout rate on classification head
    device      : "cuda" / "cpu"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TrafficSignClassifier(num_classes, backbone, pretrained, dropout)
    model = model.to(device)

    print(f"\n  [model] {backbone}  |  classes={num_classes}  "
          f"|  total params={model.total_params():,}  |  device={device}")

    return model


def save_checkpoint(model, optimizer, epoch, val_acc, config_name, label_map, checkpoint_dir=CHECKPOINT_DIR):
    """Save model + optimiser state to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{config_name}_best.pth")
    torch.save({
        "epoch":       epoch,
        "val_acc":     val_acc,
        "config_name": config_name,
        "label_map":   label_map,
        "backbone":    model.backbone_name,
        "num_classes": model.num_classes,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)
    print(f"  [checkpoint] Saved => {path}  (val_acc={val_acc:.4f})")
    return path


def load_checkpoint(path, device=None):
    """
    Loads a checkpoint
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    print(f"  [checkpoint] Loaded '{path}'  "
          f"(epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    return ckpt


def restore_model(path, device=None):
    """
    Convenience: load a checkpoint and return a ready-to-use model + label_map.

    Returns
    -------
    model      : TrafficSignClassifier  (eval mode, on device)
    label_map  : dict {class_name: int}
    """
    ckpt   = load_checkpoint(path, device)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        num_classes=ckpt["num_classes"],
        backbone=ckpt["backbone"],
        pretrained=False,
        device=device,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt["label_map"]
