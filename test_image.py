"""
╔══════════════════════════════════════════════════════════════╗
║   Driver Drowsiness & Distraction Detection                  ║
║   TEST IMAGE — Single image prediction                       ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    python test_image.py --image path/to/image.jpg
"""

import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DEVICE, CHECKPOINT_DIR, PHASE2,
    SEQ_LEN, IMG_SIZE, CLASS_NAMES,
)
from model import build_model


# ──────────────────────────────────────────────
# Transform
# ──────────────────────────────────────────────

def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


# ──────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────

def predict_image(image_path: str):
    # ── Load model ────────────────────────────
    ckpt_path = str(CHECKPOINT_DIR / PHASE2["checkpoint_name"])
    model = build_model(
        phase           = "finetune",
        checkpoint_path = ckpt_path,
        device          = DEVICE,
    )
    model.eval()

    # ── Load & preprocess image ───────────────
    img       = Image.open(image_path).convert("RGB")
    transform = get_transform()
    tensor    = transform(img)                          # (3, H, W)

    # Repeat single image SEQ_LEN times to fill sequence
    # (model expects a temporal sequence — we simulate it with one frame)
    seq = tensor.unsqueeze(0).repeat(SEQ_LEN, 1, 1, 1) # (T, 3, H, W)
    seq = seq.unsqueeze(0).to(DEVICE)                   # (1, T, 3, H, W)

    # ── Inference ─────────────────────────────
    with torch.no_grad():
        logits = model(seq)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()

    # ── Results ───────────────────────────────
    class_name = CLASS_NAMES[pred]

    print("\n" + "=" * 40)
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {class_name.upper()}")
    print(f"  Confidence : {conf * 100:.1f}%")
    print("=" * 40)
    print(f"  All class probabilities:")
    for cls_id, cls_name in CLASS_NAMES.items():
        bar = "█" * int(probs[cls_id].item() * 30)
        print(f"    {cls_name:>12s}: {probs[cls_id].item()*100:5.1f}%  {bar}")
    print("=" * 40 + "\n")

    # ── Visualize ─────────────────────────────
    color_map = {
        "safe"       : "green",
        "distracted" : "orange",
        "drowsy"     : "red",
    }
    color = color_map.get(class_name, "gray")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left — image with prediction
    axes[0].imshow(img)
    axes[0].set_title(
        f"{class_name.upper()}  —  {conf*100:.1f}%",
        fontsize=16, fontweight="bold", color=color
    )
    axes[0].axis("off")

    # Right — probability bar chart
    names  = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    values = [probs[i].item() * 100 for i in range(len(CLASS_NAMES))]
    colors = [color_map.get(n, "gray") for n in names]

    bars = axes[1].barh(names, values, color=colors, edgecolor="black")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence (%)", fontsize=12)
    axes[1].set_title("Class Probabilities", fontsize=14)

    for bar, val in zip(bars, values):
        axes[1].text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=11
        )

    plt.tight_layout()
    plt.savefig("test_image_result.png", dpi=150)
    print("💾 Result saved → test_image_result.png")
    plt.show()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image"
    )
    args = parser.parse_args()
    predict_image(args.image)