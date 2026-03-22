"""
╔══════════════════════════════════════════════════════════════╗
║   Driver Drowsiness & Distraction Detection                  ║
║   TRAINING LOOP — Phase 1 (State Farm) + Phase 2 (DDD)      ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    # Phase 1 — train on State Farm
    python train.py --phase 1

    # Phase 2 — fine-tune on DDD (requires Phase 1 checkpoint)
    python train.py --phase 2

    # Quick smoke test (3 epochs, tiny batch)
    python train.py --phase 1 --smoke
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DEVICE, NUM_THREADS, SEED,
    CHECKPOINT_DIR, LOG_DIR,
    PHASE1, PHASE2, NUM_CLASSES,
)
from data_pipeline import (
    run_phase1_pipeline,
    run_phase2_pipeline,
    DriverSequenceDataset,
)
from model import build_model, get_loss_fn

# ── Reproducibility ───────────────────────────
torch.manual_seed(SEED)
torch.set_num_threads(NUM_THREADS)


# ──────────────────────────────────────────────
# SECTION 1 — Metrics Helper
# ──────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ──────────────────────────────────────────────
# SECTION 2 — Single Epoch: Train
# ──────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    loss_fn   : nn.Module,
    device    : str,
    epoch     : int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Runs one full training epoch.
    Returns (avg_loss, avg_accuracy).
    """
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss   = loss_fn(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(logits, labels)
        n_batches  += 1

        # Progress print every 20 batches
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"  Epoch {epoch}/{total_epochs} "
                f"[{batch_idx+1}/{len(loader)}] "
                f"loss: {total_loss/n_batches:.4f}  "
                f"acc: {total_acc/n_batches:.4f}"
            )

    return total_loss / n_batches, total_acc / n_batches


# ──────────────────────────────────────────────
# SECTION 3 — Single Epoch: Validate
# ──────────────────────────────────────────────

def validate(
    model   : nn.Module,
    loader  : DataLoader,
    loss_fn : nn.Module,
    device  : str,
) -> tuple[float, float]:
    """
    Runs validation. No gradients, no augmentation.
    Returns (avg_loss, avg_accuracy).
    """
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)
            logits = model(frames)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()
            total_acc  += compute_accuracy(logits, labels)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ──────────────────────────────────────────────
# SECTION 4 — Checkpoint Helpers
# ──────────────────────────────────────────────

def save_checkpoint(
    model      : nn.Module,
    optimizer  : torch.optim.Optimizer,
    epoch      : int,
    val_loss   : float,
    val_acc    : float,
    filename   : str,
):
    path = CHECKPOINT_DIR / filename
    torch.save({
        "epoch"              : epoch,
        "model_state_dict"   : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss"           : val_loss,
        "val_acc"            : val_acc,
    }, path)
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, filename, device):
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        print(f"  No checkpoint found at {path}, starting fresh.")
        return 0, float("inf")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"  ✅ Resumed from {path} (epoch {state['epoch']})")
    return state["epoch"], state["val_loss"]


# ──────────────────────────────────────────────
# SECTION 5 — CSV Logger
# ──────────────────────────────────────────────

class CSVLogger:
    """Logs epoch metrics to a CSV file in logs/."""

    def __init__(self, filename: str):
        self.path = LOG_DIR / filename
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "val_loss", "val_acc", "lr", "epoch_time_s"
            ])

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, t):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}",   f"{val_acc:.6f}",
                f"{lr:.2e}", f"{t:.1f}",
            ])


# ──────────────────────────────────────────────
# SECTION 6 — Main Training Loop
# ──────────────────────────────────────────────

def train(
    phase      : int  = 1,
    smoke      : bool = False,
    resume     : bool = False,
):
    """
    Full training loop for Phase 1 or Phase 2.

    phase=1 : Train CNN+LSTM on State Farm (CNN frozen)
    phase=2 : Fine-tune on DDD (last 2 CNN blocks unfrozen)
    smoke   : Run 2 epochs on tiny data to verify pipeline
    resume  : Resume from last checkpoint
    """
    cfg = PHASE1 if phase == 1 else PHASE2
    print("=" * 60)
    print(f"TRAINING — Phase {phase}  |  device: {DEVICE}")
    print("=" * 60)

    # ── Data ──────────────────────────────────
    if smoke:
        # Smoke test: use pipeline but override epochs + batch
        from data_pipeline import smoke_test
        loaders    = smoke_test(batch_size=2)
        epochs     = 2
        batch_size = 2
    elif phase == 1:
        loaders    = run_phase1_pipeline(batch_size=cfg["batch_size"])
        epochs     = cfg["epochs"]
    else:
        loaders    = run_phase2_pipeline(batch_size=cfg["batch_size"])
        epochs     = cfg["epochs"]

    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── Class weights (handles imbalance) ─────
    class_weights = train_loader.dataset.get_class_weights()
    print(f"\n  Class weights: {class_weights.tolist()}")

    # ── Model ─────────────────────────────────
    if phase == 1:
        model = build_model(phase="pretrain", device=DEVICE)
    else:
        phase1_ckpt = str(CHECKPOINT_DIR / PHASE1["checkpoint_name"])
        model = build_model(
            phase="finetune",
            checkpoint_path=phase1_ckpt,
            device=DEVICE,
        )

    # ── Optimizer ─────────────────────────────
    if phase == 1:
        # Single LR — CNN is frozen, only LSTM + FC + projection train
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = cfg["learning_rate"],
            weight_decay = cfg["weight_decay"],
        )
    else:
        # Differential LR — CNN layers get much lower LR than LSTM
        cnn_params  = list(model.cnn.parameters())
        rest_params = [
            p for p in model.parameters()
            if not any(p is cp for cp in cnn_params)
        ]
        optimizer = torch.optim.Adam([
            {"params": cnn_params,  "lr": cfg["learning_rate_cnn"]},
            {"params": rest_params, "lr": cfg["learning_rate_lstm"]},
        ], weight_decay=cfg["weight_decay"])

    # ── Scheduler ─────────────────────────────
    # Reduces LR when val_loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        patience = cfg["lr_patience"],
        factor   = cfg["lr_factor"],
    )

    # ── Loss ──────────────────────────────────
    loss_fn = get_loss_fn(
        class_weights   = class_weights,
        label_smoothing = cfg["label_smoothing"],
        device          = DEVICE,
    )

    # ── Logger ────────────────────────────────
    logger = CSVLogger(cfg["log_name"])

    # ── Resume ────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    if resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, cfg["checkpoint_name"], DEVICE
        )

    # ── Early Stopping State ───────────────────
    patience_counter = 0
    early_stop       = cfg["early_stop_patience"]

    # ── Training Loop ─────────────────────────
    print(f"\n  Starting training: {epochs} epochs, "
          f"batch={cfg['batch_size']}, "
          f"early_stop patience={early_stop}\n")

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            DEVICE, epoch, epochs,
        )
        val_loss, val_acc = validate(model, val_loader, loss_fn, DEVICE)

        elapsed = time.time() - t0

        # Current LR (first param group)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to CSV
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed)

        # Print epoch summary
        print(
            f"\n  ── Epoch {epoch:03d}/{epochs} ──────────────────────\n"
            f"  train loss: {train_loss:.4f}  train acc: {train_acc:.4f}\n"
            f"  val   loss: {val_loss:.4f}  val   acc: {val_acc:.4f}\n"
            f"  lr: {current_lr:.2e}  time: {elapsed:.1f}s\n"
        )

        # LR scheduler step
        scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                val_loss, val_acc,
                cfg["checkpoint_name"],
            )
        else:
            patience_counter += 1
            print(f"  No improvement. Early stop counter: "
                  f"{patience_counter}/{early_stop}")

        # Early stopping
        if patience_counter >= early_stop:
            print(f"\n  🛑 Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n✅ Training complete.")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Checkpoint    : {CHECKPOINT_DIR / cfg['checkpoint_name']}")
    print(f"   Log           : {LOG_DIR / cfg['log_name']}")


# ──────────────────────────────────────────────
# SECTION 7 — Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train driver behavior model")
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Training phase: 1=State Farm pretrain, 2=DDD finetune"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: run 2 epochs to verify pipeline"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    args = parser.parse_args()

    train(phase=args.phase, smoke=args.smoke, resume=args.resume)