"""
╔══════════════════════════════════════════════════════════════╗
║   Driver Drowsiness & Distraction Detection                  ║
║   DATA PIPELINE — Phase 1 (State Farm) + Phase 2 (DDD)      ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    python data_pipeline.py     ← runs smoke test, no data needed

TENSOR SHAPE PRODUCED:
    (Batch, Time, Channels, Height, Width)
    e.g. (8, 10, 3, 224, 224)
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    STATEFARM_DIR, DDD_DIR,
    SEQ_LEN, IMG_SIZE, NUM_CLASSES,
    SF_VAL_RATIO, SF_TEST_RATIO, SF_SEED,
    DDD_STRIDE, DDD_SEED,
    PHASE1, PHASE2,
)


# ──────────────────────────────────────────────
# SECTION 1 — Download Datasets from Kaggle
# ──────────────────────────────────────────────

def download_datasets():
    """
    Downloads both datasets from Kaggle.
    Requires ~/.kaggle/kaggle.json

    Setup:
        pip install kaggle
        mkdir -p ~/.kaggle
        cp kaggle.json ~/.kaggle/
        chmod 600 ~/.kaggle/kaggle.json
    """
    os.makedirs(STATEFARM_DIR, exist_ok=True)
    os.makedirs(DDD_DIR, exist_ok=True)

    print("📥 Downloading State Farm Distracted Driver Detection...")
    os.system(
        f"kaggle competitions download -c state-farm-distracted-driver-detection "
        f"-p {STATEFARM_DIR} --unzip"
    )

    print("📥 Downloading Driver Drowsiness Dataset (DDD)...")
    os.system(
        f"kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd "
        f"-p {DDD_DIR} --unzip"
    )
    print("✅ Downloads complete.")


# ──────────────────────────────────────────────
# SECTION 2 — Explore State Farm Structure
# ──────────────────────────────────────────────

def explore_statefarm():
    """
    Verifies State Farm structure and returns the label CSV + image root.

    Expected structure after unzip:
        data/statefarm/
        ├── imgs/
        │   └── train/
        │       ├── c0/   ← safe driving
        │       ├── c1/   ← texting right
        │       └── ...   ← c2–c9
        └── driver_imgs_list.csv
    """
    csv_path = STATEFARM_DIR / "driver_imgs_list.csv"
    img_root = STATEFARM_DIR / "imgs" / "train"

    assert csv_path.exists(), f"CSV not found at {csv_path}"
    assert img_root.exists(), f"Train images not found at {img_root}"

    df = pd.read_csv(csv_path)
    print(f"Total labeled images : {len(df)}")
    print(f"Unique drivers       : {df['subject'].nunique()}")
    print(f"Classes              : {sorted(df['classname'].unique())}")
    print(df.head())
    return df, img_root


# ──────────────────────────────────────────────
# SECTION 3 — Class Label Mappings
# ──────────────────────────────────────────────

# 10 State Farm classes → 3 coarse labels
COARSE_LABEL_MAP = {
    "c0": 0,  # safe
    "c1": 1,  # distracted
    "c2": 1,
    "c3": 1,
    "c4": 1,
    "c5": 1,
    "c6": 1,
    "c7": 1,
    "c8": 1,
    "c9": 1,
    # label 2 (drowsy) comes from DDD dataset only
}

COARSE_NAMES = {0: "safe", 1: "distracted", 2: "drowsy"}


# ──────────────────────────────────────────────
# SECTION 4 — State Farm Pseudo-Sequence Builder
# ──────────────────────────────────────────────

def build_statefarm_sequences(df: pd.DataFrame, img_root: Path) -> list[dict]:
    """
    Converts State Farm image dataset into pseudo-video sequences.

    Strategy:
        - Group images by (subject, classname) — same driver, same behavior
        - Shuffle within each group then chunk into non-overlapping windows
        - Each window of SEQ_LEN frames gets the group's class label

    This simulates temporal data so the LSTM sees behavior evolving over time.
    """
    random.seed(SF_SEED)

    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["subject"], row["classname"])
        img_path = img_root / row["classname"] / row["img"]
        groups[key].append(img_path)

    sequences = []
    skipped = 0

    for (subject, classname), paths in groups.items():
        if len(paths) < SEQ_LEN:
            skipped += 1
            continue

        random.shuffle(paths)
        label = COARSE_LABEL_MAP[classname]
        num_seqs = len(paths) // SEQ_LEN

        for i in range(num_seqs):
            chunk = paths[i * SEQ_LEN : (i + 1) * SEQ_LEN]
            sequences.append({
                "frames" : chunk,
                "label"  : label,
                "class"  : classname,
                "subject": subject,
                "source" : "statefarm",
            })

    print(f"✅ State Farm sequences: {len(sequences)}  (skipped {skipped} small groups)")
    _print_label_dist(sequences)
    return sequences


def _print_label_dist(sequences: list[dict]):
    counts = defaultdict(int)
    for s in sequences:
        counts[s["label"]] += 1
    for lbl, n in sorted(counts.items()):
        print(f"   Label {lbl} ({COARSE_NAMES.get(lbl,'?'):>12s}): {n} sequences")

def balance_sequences(sequences: list[dict]) -> list[dict]:
    by_label = defaultdict(list)
    for s in sequences:
        by_label[s["label"]].append(s)

    random.seed(None)  # remove fixed seed so repeats aren't identical each epoch
    max_count = max(len(v) for v in by_label.values())
    balanced = []
    for label, seqs in by_label.items():
        oversampled = seqs * (max_count // len(seqs)) + random.sample(seqs, max_count % len(seqs))
        balanced.extend(oversampled)

    random.shuffle(balanced)
    print(f"✅ Balanced dataset: {len(balanced)} sequences")
    _print_label_dist(balanced)
    return balanced
# ──────────────────────────────────────────────
# SECTION 5 — Subject-Level Train/Val/Test Split
# ──────────────────────────────────────────────

def split_sequences_by_subject(
    sequences  : list[dict],
    val_ratio  : float = SF_VAL_RATIO,
    test_ratio : float = SF_TEST_RATIO,
) -> tuple[list, list, list]:
    """
    Splits sequences at driver (subject) level — not randomly.

    WHY subject-level:
        Random split leaks data — the same driver appears in train AND val,
        inflating accuracy. Subject-level split forces generalization to
        completely unseen drivers.
    """
    random.seed(SF_SEED)

    subjects = list(set(s["subject"] for s in sequences))
    random.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_ratio))
    n_val  = max(1, int(len(subjects) * val_ratio))

    test_subj  = set(subjects[:n_test])
    val_subj   = set(subjects[n_test : n_test + n_val])
    train_subj = set(subjects[n_test + n_val :])

    train = [s for s in sequences if s["subject"] in train_subj]
    val   = [s for s in sequences if s["subject"] in val_subj]
    test  = [s for s in sequences if s["subject"] in test_subj]

    print(f"\n📊 Subject-level split:")
    print(f"   Train : {len(train_subj):2d} drivers → {len(train):4d} sequences")
    print(f"   Val   : {len(val_subj):2d} drivers → {len(val):4d} sequences")
    print(f"   Test  : {len(test_subj):2d} drivers → {len(test):4d} sequences")

    return train, val, test


# ──────────────────────────────────────────────
# SECTION 6 — DDD Frame Extractor
# ──────────────────────────────────────────────

def build_ddd_sequences() -> list[dict]:
    """
    Builds sequences from DDD dataset (real drowsiness videos).

    Expected structure:
        data/ddd/
        ├── Drowsy/
        └── Non_Drowsy/  (or Alert / Awake)

    Uses sliding window with DDD_STRIDE overlap to maximize sequences
    from limited real-world video data.
    """
    random.seed(DDD_SEED)

    subdirs = [d for d in DDD_DIR.iterdir() if d.is_dir()]
    print(f"DDD subdirectories: {[d.name for d in subdirs]}")

    label_map = {}
    for d in subdirs:
        name = d.name.lower()
        if name == "drowsy":
            label_map[d] = 2
        elif "non" in name:
            label_map[d] = 0

    if not label_map:
        raise ValueError(
            f"Could not detect DDD label folders in {DDD_DIR}\n"
            f"Found: {[d.name for d in subdirs]}\n"
            "Rename folders to contain 'drowsy'/'alert' or update label_map manually."
        )

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    sequences  = []

    for folder, label in label_map.items():
        frame_files = sorted([
            f for f in folder.rglob("*")
            if f.suffix.lower() in VALID_EXTS
        ])
        if len(frame_files) < SEQ_LEN:
            continue
        for start in range(0, len(frame_files) - SEQ_LEN + 1, DDD_STRIDE):
            chunk = frame_files[start : start + SEQ_LEN]
            sequences.append({
                "frames" : chunk,
                "label"  : label,
                "class"  : "drowsy" if label == 2 else "safe",
                "subject": folder.name,
                "source" : "ddd",
            })

    print(f"✅ DDD sequences: {len(sequences)}")
    _print_label_dist(sequences)
    return sequences


# ──────────────────────────────────────────────
# SECTION 7 — Transforms
# ──────────────────────────────────────────────

def get_transforms(split: str = "train"):
    """
    ImageNet normalization because MobileNetV2 was pretrained on ImageNet.
    Train gets augmentation. Val/test get only resize + normalize.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ──────────────────────────────────────────────
# SECTION 8 — Dataset Class
# ──────────────────────────────────────────────

class DriverSequenceDataset(Dataset):
    """
    PyTorch Dataset for driver behavior sequences.

    Each item:
        frames_tensor : (SEQ_LEN, 3, H, W)
        label         : int scalar

    DataLoader batches into: (batch_size, SEQ_LEN, 3, H, W)
    """

    def __init__(self, sequences: list[dict], transform=None):
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq    = self.sequences[idx]
        label  = seq["label"]
        frames = []
        for fpath in seq["frames"]:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.stack(frames, dim=0), label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted loss — always returns NUM_CLASSES weights."""
        counts = defaultdict(int)
        for s in self.sequences:
            counts[s["label"]] += 1
        total = sum(counts.values())
        weights = torch.ones(NUM_CLASSES)   # default weight=1 for unseen classes
        for cls, cnt in counts.items():
            weights[cls] = total / (NUM_CLASSES * cnt)
        return weights


# ──────────────────────────────────────────────
# SECTION 9 — DataLoader Factory
# ──────────────────────────────────────────────

def create_dataloaders(
    train_seqs : list[dict],
    val_seqs   : list[dict],
    test_seqs  : list[dict],
    batch_size : int = PHASE1["batch_size"],
    num_workers: int = PHASE1["num_workers"],
) -> dict[str, DataLoader]:
    """Returns train/val/test DataLoaders. pin_memory=False for CPU."""
    datasets = {
        "train": DriverSequenceDataset(train_seqs, transform=get_transforms("train")),
        "val"  : DriverSequenceDataset(val_seqs,   transform=get_transforms("val")),
        "test" : DriverSequenceDataset(test_seqs,  transform=get_transforms("test")),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = False,
            drop_last   = (split == "train"),
        )

    print("\n📦 DataLoaders ready:")
    for split, loader in loaders.items():
        print(f"   {split:5s}: {len(loader.dataset):5d} sequences → "
              f"{len(loader)} batches of {batch_size}")
    return loaders


# ──────────────────────────────────────────────
# SECTION 10 — Sanity Check
# ──────────────────────────────────────────────

def verify_dataloader(loader: DataLoader, split: str = "train"):
    frames, labels = next(iter(loader))
    print(f"\n🔍 Batch check ({split}):")
    print(f"   frames shape : {frames.shape}")
    print(f"   labels       : {labels.tolist()}")
    print(f"   pixel range  : [{frames.min():.2f}, {frames.max():.2f}]")
    assert frames.ndim == 5,             "Expected (B, T, C, H, W)"
    assert frames.shape[2] == 3,         "Expected 3 channels"
    assert frames.shape[1] == SEQ_LEN,   f"Expected T={SEQ_LEN}"
    print("   ✅ Shape check passed.")


# ──────────────────────────────────────────────
# SECTION 11 — Full Pipeline Runners
# ──────────────────────────────────────────────

def run_phase1_pipeline(batch_size: int = PHASE1["batch_size"]):
    """State Farm → DataLoaders for Phase 1 training."""
    print("=" * 55)
    print("PHASE 1 — State Farm Pipeline")
    print("=" * 55)
    df, img_root = explore_statefarm()
    sequences    = build_statefarm_sequences(df, img_root)
    sequences = balance_sequences(sequences)
    train, val, test = split_sequences_by_subject(sequences)
    loaders      = create_dataloaders(train, val, test, batch_size=batch_size)
    verify_dataloader(loaders["train"])
    return loaders


def run_phase2_pipeline(batch_size: int = PHASE2["batch_size"]):
    print("=" * 55)
    print("PHASE 2 — Combined Fine-Tuning Pipeline")
    print("=" * 55)

    # Load State Farm (safe + distracted)
    df, img_root = explore_statefarm()
    sf_sequences = build_statefarm_sequences(df, img_root)
    sf_sequences = balance_sequences(sf_sequences)

    # Load DDD (safe + drowsy)
    ddd_sequences = build_ddd_sequences()

    # Combine
    import random
    all_sequences = sf_sequences + ddd_sequences
    random.shuffle(all_sequences)

    n     = len(all_sequences)
    train = all_sequences[:int(n * 0.75)]
    val   = all_sequences[int(n * 0.75):int(n * 0.90)]
    test  = all_sequences[int(n * 0.90):]

    loaders = create_dataloaders(train, val, test, batch_size=batch_size)
    verify_dataloader(loaders["train"])
    return loaders


# ──────────────────────────────────────────────
# SMOKE TEST — no real data needed
# ──────────────────────────────────────────────

def smoke_test(batch_size: int = 4):
    import tempfile
    print("🧪 Running smoke test with dummy data...")

    tmpdir = Path(tempfile.mkdtemp())  # manually managed, not auto-deleted
    dummy_seqs = []

    for i in range(40):
        frame_paths = []
        for j in range(SEQ_LEN):
            fpath = tmpdir / f"img_{i}_{j}.jpg"
            arr = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(fpath)
            frame_paths.append(fpath)

        dummy_seqs.append({
            "frames" : frame_paths,
            "label"  : random.randint(0, NUM_CLASSES - 1),
            "class"  : "test",
            "subject": f"driver_{i % 5}",
            "source" : "dummy",
        })

    train, val, test = split_sequences_by_subject(
        dummy_seqs, val_ratio=0.2, test_ratio=0.1
    )
    loaders = create_dataloaders(
        train, val, test, batch_size=batch_size, num_workers=0
    )
    verify_dataloader(loaders["train"])
    print("✅ Smoke test passed!\n")
    return loaders


if __name__ == "__main__":
    run_phase1_pipeline()