# Driver Drowsiness & Distraction Detection
### CNN + LSTM with Temporal Attention | MobileNetV2 | Real-time Webcam Inference

A deep learning system that detects driver behavior in real-time using a two-phase training pipeline. The model classifies each moment as **safe**, **distracted**, or **drowsy** by analyzing sequences of video frames rather than individual images.

---

## Table of Contents
- [Datasets](#datasets)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Limitations](#limitations)
- [Project Structure](#project-structure)

---

## Datasets

### 1. State Farm Distracted Driver Detection
- **Source**: Kaggle Competition
- **Size**: 22,424 labeled images across 26 unique drivers
- **Classes**: 10 fine-grained distraction categories (texting, phone call, eating, reaching back, etc.)
- **Usage**: Remapped to 2 coarse labels — `safe (c0)` and `distracted (c1–c9)`
- **Purpose**: Teaches the model what safe vs. distracted driving looks like

### 2. Driver Drowsiness Dataset (DDD)
- **Source**: Kaggle (ismailnasri20/driver-drowsiness-dataset-ddd)
- **Size**: 13,926 sequences across two folders — `Drowsy` and `Non Drowsy`
- **Classes**: `safe` and `drowsy`
- **Purpose**: Introduces the drowsy class with real video-like frame sequences

Neither dataset alone covers all 3 classes. State Farm has no drowsy data. DDD has no distracted data. This is why a two-phase training approach was necessary.

---

## Data Pipeline

### State Farm — Pseudo-Sequence Builder
State Farm provides individual images, not video. To feed them into an LSTM, images are grouped by `(driver, class)` and chunked into non-overlapping windows of `SEQ_LEN=10` frames. This simulates a temporal sequence where all frames show the same behavior.

**Subject-level split** is used instead of random split — val and test sets contain completely different drivers from training. This prevents data leakage where the model memorizes specific people rather than general behaviors.

```
Train : 21 drivers
Val   :  3 drivers
Test  :  2 drivers
```

### Class Balancing
State Farm has a severe imbalance — 240 safe sequences vs 1,898 distracted sequences. Without correction, the model simply predicts "distracted" for everything and achieves 89% accuracy without learning anything useful.

**Fix**: Oversample the minority class (safe) with `random.seed(None)` so repeated sequences are shuffled differently each epoch, preventing pure memorization.

### DDD — Sliding Window Builder
DDD images are already sorted in temporal order within folders. A sliding window of `SEQ_LEN=10` frames with `stride=3` is applied to generate overlapping sequences, maximizing the number of training samples from limited real-world data.

### Combined Pipeline (Phase 2)
Phase 2 merges both datasets so all 3 classes are present during fine-tuning:
```
State Farm (safe + distracted) + DDD (safe + drowsy) → combined training set
```

---

## Model Architecture

```
Input  (B, T, 3, 224, 224)
    ↓
MobileNetV2 Extractor     per-frame spatial features    (B*T, 1280)
    ↓
Linear Projection         dimensionality reduction      (B*T, 512)
    ↓
Reshape                   restore time axis             (B, T, 512)
    ↓
LSTM (2 layers)           temporal sequence modeling    (B, T, 512)
    ↓
Temporal Attention        focus on key frames           (B, 512)
    ↓
Classifier (FC layers)    final prediction              (B, 3)
```

### MobileNetV2 — CNN Backbone
MobileNetV2 was chosen as the CNN backbone for the following reasons:
- Lightweight and fast — suitable for CPU inference
- Pretrained on ImageNet — strong general visual features out of the box
- Depthwise separable convolutions — efficient spatial feature extraction
- Output: 1280-dimensional feature vector per frame

The classifier head is removed. Only the feature extractor is kept.

### Linear Projection (1280 → 512)
Before feeding CNN features to the LSTM, a linear layer projects them from 1280 to 512 dimensions. This reduces the number of LSTM parameters and speeds up convergence without significant information loss.

### LSTM — Temporal Modeling
A 2-layer LSTM processes the sequence of frame features. The LSTM captures how the driver's behavior evolves over time — something a single-frame CNN cannot do. For example, a drowsy driver's head gradually droops across frames. A single frame of head-down could be anything, but the motion trajectory is what reveals drowsiness.

- `hidden_size = 512`
- `num_layers = 2`
- `dropout = 0.5` between layers
- Forget gate bias initialized to 1.0 for better long-term memory retention

### Temporal Attention
Instead of using only the final LSTM hidden state, temporal attention computes a weighted sum over all timesteps. This lets the model focus on the most informative frames in the sequence — for example, the exact moment eyes start closing — rather than treating the last frame as most important.

```python
weights = softmax(Linear(lstm_out))   # (B, T, 1)
context = (weights * lstm_out).sum()  # (B, hidden)
```

### Classifier Head
Two fully connected layers with dropout convert the attended hidden state to class logits:
```
FC(512 → 128) → ReLU → Dropout → FC(128 → 3)
```

**Total parameters: 7,148,804**

---

## Training Strategy

### Phase 1 — Pretraining on State Farm
- **Dataset**: State Farm (safe + distracted)
- **CNN**: Fully frozen — only LSTM, projection, and classifier train
- **Optimizer**: Adam, `lr=3e-4`, `weight_decay=1e-4`
- **Scheduler**: ReduceLROnPlateau, halves LR when val loss plateaus
- **Label smoothing**: 0.2 — prevents overconfident predictions
- **Early stopping**: patience=6 epochs
- **Epochs**: 15 max (stopped at 8)

Freezing the CNN in Phase 1 is intentional — MobileNetV2 already has strong ImageNet features. Training the LSTM first on top of frozen features is faster and more stable than training everything end-to-end from scratch.

### Phase 2 — Fine-tuning on Combined Dataset
- **Dataset**: State Farm + DDD (all 3 classes)
- **CNN**: Last 4 blocks unfrozen — adapts to driver-specific features
- **Optimizer**: Adam with differential learning rates
  - CNN layers: `lr=1e-5` (much lower — avoid destroying pretrained features)
  - LSTM + FC: `lr=1e-4`
- **Label smoothing**: 0.05 — reduced for fine-tuning precision
- **Early stopping**: patience=4 epochs
- **Epochs**: 10 max (ran all 10)

Fine-tuning was necessary because Phase 1 only sees safe and distracted. Without Phase 2 on the DDD dataset, the model has no concept of drowsiness and defaults to predicting distracted for everything.

Unfreezing 4 CNN blocks (instead of all) is a deliberate tradeoff — the early layers of MobileNetV2 detect low-level features (edges, textures) that are universal, so there's no reason to retrain them. Only the deeper, more semantic layers need to adapt.

---

## Evaluation

Run evaluation on the held-out test set:

```bash
python evaluate.py --phase 2
```

Outputs a per-class classification report and saves a confusion matrix to `logs/confusion_matrix_phase2.png`.

### Results

```
              precision    recall  f1-score   support
        safe     0.9988    1.0000    0.9994       826
  distracted     1.0000    1.0000    1.0000       185
      drowsy     1.0000    0.9987    0.9993       762

    accuracy                         0.9994      1773
   macro avg     0.9996    0.9996    0.9996      1773
weighted avg     0.9994    0.9994    0.9994      1773
```

**99.94% accuracy** on the held-out test set. Only 1 sample misclassified (drowsy predicted as safe).

---

## Inference

### Real-time Webcam
```bash
python inference.py
```
- Buffers 10 frames, runs CNN+LSTM+Attention, displays prediction live
- Color coded: green (safe), orange (distracted), red (drowsy)
- Drowsy alert banner appears at bottom when confidence ≥ 65%
- Press `Q` to quit

### Single Image Test
```bash
python test_image.py --image path/to/image.jpg
```
- Repeats the image SEQ_LEN times to fill the sequence buffer
- Outputs prediction + probability bar chart
- Saves result to `test_image_result.png`

> **Note**: Single image testing has reduced accuracy because the LSTM receives no real temporal information. The webcam inference is the intended evaluation method.

---

## Limitations

### 1. No Face Detection Preprocessing
The model sees the entire frame including background, clothing, and environment — not just the face and eyes. A proper pipeline would crop the face region using MediaPipe or OpenCV before the CNN, focusing attention on the most relevant pixels.

### 2. Domain Gap
- State Farm images were captured from dashboard-mounted cameras at a top-down angle
- DDD was recorded in controlled lab conditions with consistent lighting
- Webcam inference uses a completely different angle, distance, and lighting
- The model has never seen the user's specific face or environment during training

### 3. Temporal Inconsistency on CPU
On slow CPUs, frame capture timing is inconsistent. The 10-frame buffer may not represent a smooth 1-second window, causing the LSTM to see irregular temporal patterns it was not trained on.

### 4. Dataset Class Separation
The two datasets never overlap — State Farm has no drowsy data, DDD has no distracted data and was collected in very different conditions. The model learns the boundaries between classes from different distributions, making edge cases between drowsy and distracted ambiguous.

### 5. Oversampling Artifact
Safe sequences were repeated ~8x during Phase 1 to balance the dataset. Despite random shuffling, the model may have partially memorized certain safe frame patterns rather than learning fully generalizable features.

### 6. Resolution Trade-off
Training was done at `112×112` (reduced from `224×224`) to make CPU training feasible. Fine details like eye openness — critical for drowsiness detection — are harder to extract at lower resolution.

### 7. Single Image Limitation
Because the architecture is sequence-based, static image testing is inherently unreliable. A drowsy person in a single frame can look identical to someone leaning forward, which the model may classify as distracted.

### 8. Uncalibrated Confidence Scores
Softmax probabilities are not temperature-calibrated. A 95% confidence score does not mean the model is truly 95% certain — it only means that class scored highest relative to others.

---

## Project Structure

```
driver-drowsiness/
├── data/
│   ├── statefarm/          ← State Farm dataset
│   └── ddd/                ← Driver Drowsiness Dataset
├── checkpoints/
│   ├── phase1_best.pth     ← Best Phase 1 weights
│   └── phase2_best.pth     ← Best Phase 2 weights (final model)
├── logs/
│   ├── phase1_train.csv
│   ├── phase2_train.csv
│   └── confusion_matrix_phase2.png
├── config.py               ← All hyperparameters and paths
├── data_pipeline.py        ← Dataset loading, sequencing, balancing
├── model.py                ← CNN + LSTM + Attention architecture
├── train.py                ← Two-phase training loop
├── evaluate.py             ← Test set evaluation + confusion matrix
├── inference.py            ← Real-time webcam inference
└── test_image.py           ← Single image prediction
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/driver-drowsiness.git
cd driver-drowsiness

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python scikit-learn seaborn matplotlib pandas pillow tqdm

# Download datasets (requires Kaggle API key)
python data_pipeline.py

# Train
python train.py --phase 1
python train.py --phase 2

# Evaluate
python evaluate.py --phase 2

# Run inference
python inference.py
```