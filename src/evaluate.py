import os
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")            # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from dataset  import get_datasets, get_dataloaders, IMAGES_DIR, ANNOTATIONS_CSV
from model    import restore_model
from augmentation import get_val_transform


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = (BASE_DIR / "../dataset").resolve()

CHECKPOINT_DIR = (BASE_DIR / "../results/checkpoints").resolve()
RESULTS_DIR    = (BASE_DIR / "../results").resolve()
FIGURES_DIR    = (BASE_DIR / "../results/figures").resolve()
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_on_test(model, test_loader, class_names, device):
    """
    Run the model on the test set and collect predictions + ground truth.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    cm       = confusion_matrix(all_labels, all_preds)
    report   = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
    )

    print(f"\n  Test Accuracy : {accuracy:.4f}")
    print(f"\n  Classification Report:\n{report}")

    return {
        "all_preds":   all_preds,
        "all_labels":  all_labels,
        "accuracy":    accuracy,
        "report":      report,
        "cm":          cm,
        "class_names": class_names,
    }


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots()

    ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_training_curves(history, title="Training Curves", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    # --- Loss ---
    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    # --- Accuracy ---
    ax_acc.plot(epochs, history["train_acc"], label="Train")
    ax_acc.plot(epochs, history["val_acc"], label="Val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def compare_confusion_matrices(eval_results_list, title="Confusion Matrix Comparison", save_path=None):
    n = len(eval_results_list)

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, res in enumerate(eval_results_list):
        cm = res["cm"]
        class_names = res["class_names"]
        label = res.get("config_label", f"Config {i+1}")
        acc = res.get("accuracy", 0.0)

        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        ax = axes[i]
        ax.imshow(cm_norm, cmap="Blues")

        ax.set_title(f"{label} (acc={acc:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_misclassified(eval_result, test_ds, n_samples=16, title="Misclassified Samples", save_path=None):
    all_preds = eval_result["all_preds"]
    all_labels = eval_result["all_labels"]
    class_names = eval_result["class_names"]

    wrong_idx = np.where(all_preds != all_labels)[0]

    if len(wrong_idx) == 0:
        print("No misclassified samples found")
        return None

    sample_idx = wrong_idx[:n_samples]
    n = len(sample_idx)

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, idx in enumerate(sample_idx):
        img, _ = test_ds[idx]

        img = img.permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        true_label = class_names[all_labels[idx]]
        pred_label = class_names[all_preds[idx]]

        axes[i].imshow(img)
        axes[i].set_title(f"T: {true_label}\nP: {pred_label}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def run_full_evaluation(config_names=None, device=None):
    """
    Load each trained checkpoint, evaluate on the test set and save all plots.

    config_names : list of config names (e.g. ["config1_baseline", ...]).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if config_names is None:
        config_names = [
            "config1_baseline",
            "config2_augmented",
            "config3_synthetic",
            "config4_synth_augmented",
        ]

    val_transform = get_val_transform()

    _, _, test_ds, label_map, class_names = get_datasets(
        images_dir=IMAGES_DIR,
        annotations_csv=ANNOTATIONS_CSV,
        val_test_transform=val_transform,
    )

    # Train and validation not needed
    _, _, test_loader = get_dataloaders(test_ds, test_ds, test_ds)

    all_eval_results = []

    for cfg_name in config_names:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{cfg_name}_best.pth")
        if not Path(ckpt_path).exists():
            print(f"  [evaluate] Checkpoint not found: {ckpt_path} - skipping")
            continue

        print(f"\n{'='*55}")
        print(f"  Evaluating: {cfg_name}")
        print(f"{'='*55}")

        model, _ = restore_model(ckpt_path, device=device)

        eval_res = evaluate_on_test(model, test_loader, class_names, device)
        eval_res["config_label"] = cfg_name

        # Confusion matrix
        plot_confusion_matrix(
            eval_res["cm"], class_names,
            title=f"Confusion Matrix - {cfg_name}",
            save_path=os.path.join(FIGURES_DIR, f"{cfg_name}_cm.png"),
        )

        # Misclassified
        plot_misclassified(
            eval_res,
            test_ds,
            save_path=os.path.join(FIGURES_DIR, f"{cfg_name}_misclassified.png"),
            title=f"Misclassified Samples - {cfg_name}"
        )

        # Training curves
        history_path = os.path.join(RESULTS_DIR, f"{cfg_name}_history.json")
        if Path(history_path).exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(
                history,
                save_path=os.path.join(FIGURES_DIR, f"{cfg_name}_curves.png"),
            )

        all_eval_results.append(eval_res)

    # Side-by-side comparison of all evaluated configs
    if len(all_eval_results) >= 2:
        compare_confusion_matrices(
            all_eval_results,
            save_path=os.path.join(FIGURES_DIR, "comparison_confusion_matrices.png"),
        )

    # Print final accuracy table
    print(f"\n{'='*55}")
    print(f"  EVALUATION SUMMARY  (test set)")
    print(f"{'='*55}")
    print(f"  {'Config':<35} {'Accuracy':>10}")
    print(f"  {'-'*45}")
    for res in all_eval_results:
        print(f"  {res['config_label']:<35} {res['accuracy']:>10.4f}")
    print(f"{'='*55}\n")

    return all_eval_results


if __name__ == "__main__":
    run_full_evaluation()
