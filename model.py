"""
╔══════════════════════════════════════════════════════════════╗
║   Driver Drowsiness & Distraction Detection                  ║
║   MODEL — CNN (MobileNetV2) + LSTM                           ║
╚══════════════════════════════════════════════════════════════╝

ARCHITECTURE:
    Input  (B, T, 3, 224, 224)
    → Reshape     (B*T, 3, 224, 224)
    → MobileNetV2 (B*T, 1280)
    → Projection  (B*T, 512)
    → Reshape     (B, T, 512)
    → LSTM        (B, 512)   last hidden state
    → FC          (B, 3)     logits

USAGE:
    python model.py     ← runs forward pass smoke test
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

from config import (
    NUM_CLASSES,
    CNN_OUTPUT_DIM, PROJECTION_DIM,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_DROPOUT,
    PHASE1, PHASE2,
    DEVICE, SEQ_LEN,
)


# ──────────────────────────────────────────────
# SECTION 1 — CNN Feature Extractor
# ──────────────────────────────────────────────

class MobileNetV2Extractor(nn.Module):
    """
    MobileNetV2 backbone with the classifier head removed.
    Produces a (batch, 1280) feature vector per frame.

    Freezing strategy:
        Phase 1 → freeze ALL layers (only LSTM + FC train)
        Phase 2 → unfreeze last 2 InvertedResidual blocks
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        backbone       = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features  = backbone.features
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.output_dim = CNN_OUTPUT_DIM   # 1280

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreeze the last n feature layers for fine-tuning."""
        self.freeze_backbone()
        for layer in list(self.features.children())[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   CNN trainable params after unfreezing {n} blocks: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B*T, 3, 224, 224)
        out : (B*T, 1280)
        """
        x = self.features(x)    # (B*T, 1280, 7, 7)
        x = self.pool(x)        # (B*T, 1280, 1, 1)
        x = x.flatten(1)        # (B*T, 1280)
        return x


# ──────────────────────────────────────────────
# SECTION 2 — Full CNN + LSTM Model
# ──────────────────────────────────────────────

class DriverBehaviorModel(nn.Module):
    """
    Full CNN + LSTM model for driver behavior classification.

    Forward pass:
        1. Flatten time into batch dim     (B*T, C, H, W)
        2. CNN extracts spatial features   (B*T, 1280)
        3. Project features down           (B*T, 512)
        4. Restore time axis               (B, T, 512)
        5. LSTM over sequence              (B, hidden)  ← last hidden state
        6. Dropout + FC → logits           (B, num_classes)
    """

    def __init__(
        self,
        num_classes   : int   = NUM_CLASSES,
        hidden_size   : int   = LSTM_HIDDEN_SIZE,
        num_layers    : int   = LSTM_NUM_LAYERS,
        lstm_dropout  : float = LSTM_DROPOUT,
        fc_dropout    : float = FC_DROPOUT,
        freeze_cnn    : bool  = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # CNN backbone
        self.cnn = MobileNetV2Extractor(freeze=freeze_cnn)

        # Project 1280 → 512 before LSTM (fewer LSTM params, faster convergence)
        self.feature_proj = nn.Sequential(
            nn.Linear(CNN_OUTPUT_DIM, PROJECTION_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size   = PROJECTION_DIM,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = lstm_dropout if num_layers > 1 else 0.0,
            bidirectional= False,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """Xavier + orthogonal init; forget gate bias → 1.0 for better memory."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)  # forget gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, T, C, H, W)
        returns: (B, num_classes) logits
        """
        B, T, C, H, W = x.shape

        # CNN: process all frames at once
        x_flat   = x.view(B * T, C, H, W)
        cnn_grad = any(p.requires_grad for p in self.cnn.parameters())
        with torch.set_grad_enabled(cnn_grad):
            features = self.cnn(x_flat)             # (B*T, 1280)

        # Project + restore time axis
        features = self.feature_proj(features)      # (B*T, 512)
        features = features.view(B, T, -1)          # (B, T, 512)

        # LSTM: returns hidden state at every timestep
        _, (h_n, _) = self.lstm(features)           # h_n: (num_layers, B, hidden)

        # Take top layer's final hidden state
        last_hidden = h_n[-1]                       # (B, hidden)
        return self.classifier(last_hidden)         # (B, num_classes)

    def get_sequence_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-timestep logits — useful for visualizing when
        in the sequence the model first detects drowsiness/distraction.

        returns: (B, T, num_classes)
        """
        B, T, C, H, W = x.shape
        with torch.no_grad():
            features = self.cnn(x.view(B * T, C, H, W))
        features  = self.feature_proj(features).view(B, T, -1)
        lstm_out, _ = self.lstm(features)           # (B, T, hidden)
        return self.classifier(lstm_out)            # (B, T, num_classes)

    def unfreeze_cnn_for_finetuning(self, n_blocks: int = PHASE2["unfreeze_cnn_blocks"]):
        self.cnn.unfreeze_last_n_blocks(n_blocks)

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────
# SECTION 3 — Loss Function
# ──────────────────────────────────────────────

def get_loss_fn(
    class_weights   : torch.Tensor | None = None,
    label_smoothing : float = 0.1,
    device          : str   = DEVICE,
) -> nn.CrossEntropyLoss:
    """
    Weighted cross-entropy with label smoothing.
    Pass class_weights from dataset.get_class_weights() to handle imbalance.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


# ──────────────────────────────────────────────
# SECTION 4 — Model Factory
# ──────────────────────────────────────────────

def build_model(
    phase            : str        = "pretrain",
    checkpoint_path  : str | None = None,
    device           : str        = DEVICE,
) -> DriverBehaviorModel:
    """
    Builds the model for a given training phase.

    phase="pretrain"  → CNN frozen, ~1.3M trainable params
    phase="finetune"  → loads Phase 1 weights, unfreezes last 2 CNN blocks
    """
    model = DriverBehaviorModel(freeze_cnn=(phase == "pretrain"))

    if phase == "finetune":
        assert checkpoint_path, "checkpoint_path is required for finetune phase"
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"✅ Loaded Phase 1 weights from {checkpoint_path}")
        model.unfreeze_cnn_for_finetuning()

    model = model.to(device)

    p = model.count_parameters()
    print(f"\n🧠 Model ready ({phase}):")
    print(f"   Total params    : {p['total']:>10,}")
    print(f"   Trainable params: {p['trainable']:>10,}")
    print(f"   Frozen params   : {p['frozen']:>10,}")

    return model


# ──────────────────────────────────────────────
# SMOKE TEST
# ──────────────────────────────────────────────

def smoke_test(device: str = DEVICE):
    print("🧪 Testing model forward pass...")

    model = build_model(phase="pretrain", device=device)
    model.eval()

    B, T = 2, SEQ_LEN
    dummy = torch.randn(B, T, 3, 224, 224).to(device)
    lbls  = torch.randint(0, NUM_CLASSES, (B,)).to(device)

    with torch.no_grad():
        logits = model(dummy)

    print(f"   Input  : {dummy.shape}")
    print(f"   Output : {logits.shape}   (expected [{B}, {NUM_CLASSES}])")
    assert logits.shape == (B, NUM_CLASSES)

    loss_fn = get_loss_fn(device=device)
    loss    = loss_fn(logits, lbls)
    print(f"   Loss   : {loss.item():.4f}  (random init ≈ {__import__('math').log(NUM_CLASSES):.4f})")

    per_frame = model.get_sequence_predictions(dummy)
    print(f"   Per-frame logits: {per_frame.shape}   (expected [{B}, {T}, {NUM_CLASSES}])")
    assert per_frame.shape == (B, T, NUM_CLASSES)

    print("   ✅ Forward pass smoke test passed!\n")


if __name__ == "__main__":
    smoke_test()