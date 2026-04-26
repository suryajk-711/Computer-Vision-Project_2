import os
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset     import get_datasets, get_dataloaders, print_split_summary
from model       import get_model, save_checkpoint
from augmentation import get_train_transform, get_val_transform
from synthesize  import get_synthetic_rows


BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = (BASE_DIR / "../dataset").resolve()
ANNOTATIONS_CSV = (BASE_DIR / "../dataset/annotations.csv").resolve()
SYNTHETIC_DIR   = (BASE_DIR / "../dataset/synthetic").resolve()
CHECKPOINT_DIR  = (BASE_DIR / "../results/checkpoints").resolve()
RESULTS_DIR     = (BASE_DIR / "../results").resolve()

# Training hyperparameters
NUM_EPOCHS      = 30
BATCH_SIZE      = 32
RANDOM_SEED     = 42

# Phase-1: backbone frozen - higher LR on head
LR_HEAD         = 1e-3

# Phase-2: backbone unfrozen - lower LR for fine-tuning
LR_FINETUNE     = 1e-4
UNFREEZE_EPOCH  = 10        # after this epoch, unfreeze last N backbone blocks
UNFREEZE_N_BLOCKS = 3       # how many backbone blocks to unfreeze

WEIGHT_DECAY    = 1e-4
LR_STEP_SIZE    = 7         # StepLR: decay every N epochs
LR_GAMMA        = 0.5       # StepLR: multiply LR by this factor

# Early stopping
PATIENCE        = 8

# N synthetic images per class for configs 3 and 4
N_SYNTHETIC_PER_CLASS = 50

# Change it to check the configuration you want.
CONFIGS_TO_RUN = [1, 2, 3, 4]


# To Reproduce same results
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


CONFIGS = {
    1: {
        "name":            "config1_baseline",
        "description":     "Baseline - original data only, no augmentation or synthesis",
        "use_augmentation": False,
        "use_synthetic":    False,
    },
    2: {
        "name":            "config2_augmented",
        "description":     "Augmented - original data + standard augmentation",
        "use_augmentation": True,
        "use_synthetic":    False,
    },
    3: {
        "name":            "config3_synthetic",
        "description":     "Synthetic - original data + synthesized images, no augmentation",
        "use_augmentation": False,
        "use_synthetic":    True,
    },
    4: {
        "name":            "config4_synth_augmented",
        "description":     "Full - original data + synthesized + augmentation",
        "use_augmentation": True,
        "use_synthetic":    True,
    },
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0-1)
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


def validate_one_epoch(model, loader, criterion, device):
    """
    Run one validation epoch (no gradient computation).

    Returns
    -------
    avg_loss : float
    accuracy : float  (0-1)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += imgs.size(0)

    return running_loss / total, correct / total


def run_training(config_id, device):
    """
    Train the model for one configuration.

    Returns a history dict with keys:
        train_loss, train_acc, val_loss, val_acc  - list per epoch
        best_val_acc, best_epoch
        config_name, description
    """
    cfg = CONFIGS[config_id]
    set_seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"  CONFIG {config_id}: {cfg['description']}")
    print(f"{'='*60}")

    # 1. Datasets
    train_transform = get_train_transform(config_id)
    val_transform   = get_val_transform()

    # Synthetic rows (only for configs 3 and 4)
    syn_rows    = None
    syn_img_dir = IMAGES_DIR   # default - overridden below if synthetic is used

    if cfg["use_synthetic"]:
        syn_rows, syn_img_dir = get_synthetic_rows(
            images_dir=IMAGES_DIR,
            annotations_csv=ANNOTATIONS_CSV,
            output_dir=SYNTHETIC_DIR,
            n_per_class=N_SYNTHETIC_PER_CLASS,
            regenerate=False,
        )

    train_ds, val_ds, test_ds, label_map, class_names = get_datasets(
        images_dir=IMAGES_DIR,
        annotations_csv=ANNOTATIONS_CSV,
        train_transform=train_transform,
        val_test_transform=val_transform,
        synthetic_rows=syn_rows,
        synthetic_images_dir=syn_img_dir if cfg["use_synthetic"] else None,
    )

    print_split_summary(train_ds, val_ds, test_ds, class_names)

    train_loader, val_loader, _ = get_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=BATCH_SIZE
    )

    # 2. Model
    num_classes = len(class_names)
    model       = get_model(num_classes=num_classes, device=device)

    # 3. Loss, optimiser, scheduler
    # Phase 1: only head params (backbone frozen)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # 4. Training loop
    history = {
        "config_id":    config_id,
        "config_name":  cfg["name"],
        "description":  cfg["description"],
        "train_loss":   [],
        "train_acc":    [],
        "val_loss":     [],
        "val_acc":      [],
        "best_val_acc": 0.0,
        "best_epoch":   0,
        "label_map":    label_map,
        "class_names":  class_names,
    }

    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Phase 2: unfreeze backbone after UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH + 1:
            print(f"\n  [train] Epoch {epoch}: unfreezing last {UNFREEZE_N_BLOCKS} blocks")
            model.unfreeze_backbone(unfreeze_last_n_blocks=UNFREEZE_N_BLOCKS)
            # Reset optimiser with lower LR for all params
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch [{epoch:>3}/{NUM_EPOCHS}]  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"]   = epoch
            patience_count = 0

            save_checkpoint(
                model, optimizer, epoch, val_acc,
                config_name=cfg["name"],
                label_map=label_map,
                checkpoint_dir=CHECKPOINT_DIR,
            )
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  [train] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    print(f"\n  Config {config_id} done. "
          f"Best val_acc={best_val_acc:.4f} at epoch {history['best_epoch']}")

    return history


def save_history(history, results_dir=RESULTS_DIR):
    """Save training history to a JSON file for later plotting."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{history['config_name']}_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [train] History saved => {path}")
    return path


def load_history(config_name, results_dir=RESULTS_DIR):
    path = os.path.join(results_dir, f"{config_name}_history.json")
    with open(path) as f:
        return json.load(f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    set_seed(RANDOM_SEED)

    all_histories = {}

    for config_id in CONFIGS_TO_RUN:
        history = run_training(config_id, device)
        save_history(history)
        all_histories[config_id] = history

    # Summary table
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<40} {'Best Val Acc':>12}  {'Epoch':>6}")
    print(f"  {'-'*58}")
    for cid, h in all_histories.items():
        print(f"  {h['description']:<40} "
              f"{h['best_val_acc']:>12.4f}  {h['best_epoch']:>6}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()