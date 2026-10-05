import torch
from torch import nn

from age_config import NUM_AGE_CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        scale = self.pool(x).view(batch_size, channels)
        scale = self.gate(scale).view(batch_size, channels, 1, 1)
        return x * scale


class ResidualSEBlock(nn.Module):
    """Residual block with BatchNorm, channel attention and light dropout."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.feature_dropout = (
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.feature_dropout(out)

        out = out + identity
        return self.relu(out)


def make_stage(
    in_channels,
    out_channels,
    num_blocks,
    stride,
    dropout,
):
    blocks = [
        ResidualSEBlock(
            in_channels,
            out_channels,
            stride=stride,
            dropout=dropout,
        )
    ]

    for _ in range(num_blocks - 1):
        blocks.append(
            ResidualSEBlock(
                out_channels,
                out_channels,
                stride=1,
                dropout=dropout,
            )
        )

    return nn.Sequential(*blocks)


class ComplexCNN(nn.Module):
    """
    Improved multitask CNN for age-group and gender classification.

    Design choices:
    - A two-layer 3x3 stem preserves facial detail better than one aggressive
      7x7 convolution.
    - Shared residual stages learn general face features.
    - SE attention recalibrates useful channels automatically.
    - The age branch is wider because age classification is harder.
    - The gender branch is smaller to reduce unnecessary parameters and
      overfitting.
    - The model is trained from scratch and keeps the original output API.
    """

    def __init__(self, head_dropout=0.35):
        super().__init__()

        # 224x224 -> 56x56 while retaining more local information.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Shared trunk: 56x56x32 -> 14x14x256.
        self.shared_stage1 = make_stage(
            32, 64, num_blocks=2, stride=1, dropout=0.05
        )
        self.shared_stage2 = make_stage(
            64, 128, num_blocks=2, stride=2, dropout=0.05
        )
        self.shared_stage3 = make_stage(
            128, 256, num_blocks=2, stride=2, dropout=0.10
        )

        # Age receives more capacity because age groups are harder to separate.
        self.age_branch = make_stage(
            256, 256, num_blocks=2, stride=1, dropout=0.15
        )

        # Gender is a simpler task, so this branch is intentionally smaller.
        self.gender_branch = make_stage(
            256, 128, num_blocks=1, stride=2, dropout=0.10
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(128, NUM_AGE_CLASSES),
        )

        self.gender_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout * 0.75),
            nn.Linear(64, 2),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

        # Start residual blocks close to identity for more stable optimization.
        for module in self.modules():
            if isinstance(module, ResidualSEBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, images):
        x = self.stem(images)
        x = self.shared_stage1(x)
        x = self.shared_stage2(x)
        shared_features = self.shared_stage3(x)

        age_features = self.age_branch(shared_features)
        age_features = self.pool(age_features).flatten(1)

        gender_features = self.gender_branch(shared_features)
        gender_features = self.pool(gender_features).flatten(1)

        return {
            "age": self.age_head(age_features),
            "gender": self.gender_head(gender_features),
        }

    def get_gradcam_layer(self, task="age"):
        """Return the final convolution for the requested task."""
        if task == "age":
            return self.age_branch[-1].conv2
        if task == "gender":
            return self.gender_branch[-1].conv2
        raise ValueError("task must be 'age' or 'gender'")
