"""
╔══════════════════════════════════════════════════════════════╗
║   Driver Drowsiness & Distraction Detection                  ║
║   EVALUATE — Test set evaluation + confusion matrix          ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    python evaluate.py --phase 1   ← evaluate phase1_best.pth on State Farm test set
    python evaluate.py --phase 2   ← evaluate phase2_best.pth on DDD test set
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from config import (
    DEVICE, CHECKPOINT_DIR, LOG_DIR,
    PHASE1, PHASE2, NUM_CLASSES, CLASS_NAMES,
)
from data_pipeline import run_phase1_pipeline, run_phase2_pipeline
from model import build_model


# ──────────────────────────────────────────────
# SECTION 1 — Run Evaluation
# ──────────────────────────────────────────────

def evaluate(phase: int = 2):
    print("=" * 60)
    print(f"EVALUATION — Phase {phase}  |  device: {DEVICE}")
    print("=" * 60)

    # ── Data ──────────────────────────────────
    if phase == 1:
        loaders      = run_phase1_pipeline(batch_size=PHASE1["batch_size"])
        ckpt_name    = PHASE1["checkpoint_name"]
    else:
        loaders      = run_phase2_pipeline(batch_size=PHASE2["batch_size"])
        ckpt_name    = PHASE2["checkpoint_name"]

    test_loader = loaders["test"]

    # ── Model ─────────────────────────────────
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    model = build_model(
        phase      = "pretrain" if phase == 1 else "finetune",
        checkpoint_path = str(ckpt_path) if phase == 2 else None,
        device     = DEVICE,
    )

    # Load weights for phase 1 manually
    if phase == 1:
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"])
        print(f"✅ Loaded weights from {ckpt_path}")

    model.eval()

    # ── Inference ─────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(DEVICE)
            logits = model(frames)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────
    present_classes = sorted(set(all_labels.tolist()))
    target_names    = [CLASS_NAMES[i] for i in present_classes]

    print("\n📊 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        labels      = present_classes,
        target_names= target_names,
        digits      = 4,
    ))

    # ── Confusion Matrix ──────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=present_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix — Phase {phase}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    plot_path = LOG_DIR / f"confusion_matrix_phase{phase}.png"
    plt.savefig(plot_path)
    print(f"\n💾 Confusion matrix saved → {plot_path}")
    plt.show()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", type=int, default=2, choices=[1, 2],
        help="Which checkpoint to evaluate"
    )
    args = parser.parse_args()
    evaluate(phase=args.phase)