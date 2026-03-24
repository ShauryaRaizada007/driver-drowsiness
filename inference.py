"""
╔══════════════════════════════════════════════════════════════╗
║   Driver Drowsiness & Distraction Detection                  ║
║   INFERENCE — Real-time webcam prediction                    ║
╚══════════════════════════════════════════════════════════════╝

USAGE:
    python inference.py
"""

import cv2
import torch
import numpy as np
from collections import deque
from PIL import Image
import torchvision.transforms as T

from config import (
    DEVICE, CHECKPOINT_DIR, PHASE2,
    SEQ_LEN, IMG_SIZE, CLASS_NAMES,
    INFERENCE_CONFIDENCE, WEBCAM_INDEX,
)
from model import build_model


# ──────────────────────────────────────────────
# SECTION 1 — Transform (same as val/test)
# ──────────────────────────────────────────────

def get_inference_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


# ──────────────────────────────────────────────
# SECTION 2 — Display Helpers
# ──────────────────────────────────────────────

# Color per class (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 200, 0),    # safe      → green
    1: (0, 165, 255),  # distracted → orange
    2: (0, 0, 255),    # drowsy    → red
}

def draw_overlay(frame, label, confidence, buffer_size):
    h, w = frame.shape[:2]

    # Background bar
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

    # Class label + confidence
    color     = CLASS_COLORS.get(label, (255, 255, 255))
    class_str = CLASS_NAMES.get(label, "unknown").upper()
    conf_str  = f"{confidence * 100:.1f}%"

    cv2.putText(frame, class_str, (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
    cv2.putText(frame, conf_str, (w - 130, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Buffer progress bar
    bar_w = int((buffer_size / SEQ_LEN) * (w - 30))
    cv2.rectangle(frame, (15, 65), (15 + bar_w, 75), color, -1)
    cv2.rectangle(frame, (15, 65), (w - 15, 75), (100, 100, 100), 1)

    # Alert if drowsy
    if label == 2 and confidence >= INFERENCE_CONFIDENCE:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "⚠  DROWSY ALERT  ⚠", (w // 2 - 160, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return frame


# ──────────────────────────────────────────────
# SECTION 3 — Main Inference Loop
# ──────────────────────────────────────────────

from collections import Counter
prediction_history = deque(maxlen=10)  # smooth over last 10 predictions

def run_inference():
    # ── Load model ────────────────────────────
    ckpt_path = str(CHECKPOINT_DIR / PHASE2["checkpoint_name"])
    model = build_model(
        phase           = "finetune",
        checkpoint_path = ckpt_path,
        device          = DEVICE,
    )
    model.eval()
    print(f"\n✅ Model loaded from {ckpt_path}")

    transform   = get_inference_transform()
    frame_buffer = deque(maxlen=SEQ_LEN)

    # ── Webcam ────────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {WEBCAM_INDEX}")

    print(f"📷 Webcam opened. Press Q to quit.\n")

    current_label      = 0
    current_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # ── Preprocess frame ──────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        tensor = transform(pil)
        frame_buffer.append(tensor)

        # ── Predict when buffer is full ────────
        if len(frame_buffer) == SEQ_LEN:
            seq = torch.stack(list(frame_buffer), dim=0)
            seq = seq.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(seq)
                probs  = torch.softmax(logits, dim=1)[0]
                pred   = probs.argmax().item()
                conf   = probs[pred].item()

            prediction_history.append(pred)

            # Majority vote over recent predictions
            current_label      = Counter(prediction_history).most_common(1)[0][0]
            current_confidence = conf

        # ── Draw overlay ──────────────────────
        display = draw_overlay(
            frame.copy(),
            current_label,
            current_confidence,
            len(frame_buffer),
        )

        cv2.imshow("Driver Monitor", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Inference stopped.")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_inference()