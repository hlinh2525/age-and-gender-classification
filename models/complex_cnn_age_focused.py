import torch
from torch import nn

from age_config import NUM_AGE_CLASSES


class SEBlock(nn.Module):
    """Channel attention used to emphasize useful facial features."""

    def __init__(self, channels, reduction=16):
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


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm, SE attention and light dropout."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
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
        self.dropout = (
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

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)

        return self.relu(out + identity)


def make_stage(in_channels, out_channels, num_blocks, stride, dropout):
    blocks = [
        ResidualBlock(
            in_channels,
            out_channels,
            stride=stride,
            dropout=dropout,
        )
    ]

    for _ in range(num_blocks - 1):
        blocks.append(
            ResidualBlock(
                out_channels,
                out_channels,
                stride=1,
                dropout=dropout,
            )
        )

    return nn.Sequential(*blocks)


class FeatureProjection(nn.Module):
    """Project a feature map and convert it into one feature vector."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.projection(x).flatten(1)


class ComplexCNN(nn.Module):
    """
    Age-focused Complex CNN for multitask classification.

    The age head receives multi-scale features from the shared trunk:
    - early features: local facial details and texture;
    - middle features: facial parts and proportions;
    - deep features: global face structure.

    The gender head uses a separate lightweight branch. The output interface
    remains compatible with the existing training.py:

        {"age": age_logits, "gender": gender_logits}

    This model is trained from scratch and does not set any random seed.
    """

    def __init__(self, head_dropout=0.30):
        super().__init__()

        # Input: 224x224x3 -> 56x56x32.
        # Two 3x3 convolutions preserve more local information than a single
        # aggressive 7x7 convolution.
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

        # Shared feature extractor.
        # stage1: 56x56x64
        # stage2: 28x28x128
        # stage3: 14x14x256
        self.shared_stage1 = make_stage(
            32,
            64,
            num_blocks=2,
            stride=1,
            dropout=0.03,
        )
        self.shared_stage2 = make_stage(
            64,
            128,
            num_blocks=2,
            stride=2,
            dropout=0.05,
        )
        self.shared_stage3 = make_stage(
            128,
            256,
            num_blocks=2,
            stride=2,
            dropout=0.08,
        )

        # Age branch is deeper and keeps 256 channels.
        self.age_branch = make_stage(
            256,
            256,
            num_blocks=2,
            stride=1,
            dropout=0.08,
        )

        # Gender is easier and receives a smaller task-specific branch.
        self.gender_branch = make_stage(
            256,
            128,
            num_blocks=1,
            stride=2,
            dropout=0.05,
        )

        # Multi-scale age feature extraction.
        self.age_early_projection = FeatureProjection(64, 64)
        self.age_middle_projection = FeatureProjection(128, 128)
        self.age_deep_projection = FeatureProjection(256, 256)

        self.age_head = nn.Sequential(
            nn.Linear(64 + 128 + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout * 0.5),
            nn.Linear(128, NUM_AGE_CLASSES),
        )

        self.gender_pool = nn.AdaptiveAvgPool2d(1)
        self.gender_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout * 0.5),
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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

    def forward(self, images):
        x = self.stem(images)

        early_features = self.shared_stage1(x)
        middle_features = self.shared_stage2(early_features)
        deep_features = self.shared_stage3(middle_features)

        age_deep_features = self.age_branch(deep_features)

        age_features = torch.cat(
            [
                self.age_early_projection(early_features),
                self.age_middle_projection(middle_features),
                self.age_deep_projection(age_deep_features),
            ],
            dim=1,
        )

        gender_features = self.gender_branch(deep_features)
        gender_features = self.gender_pool(gender_features).flatten(1)

        return {
            "age": self.age_head(age_features),
            "gender": self.gender_head(gender_features),
        }

    def get_gradcam_layer(self, task="age"):
        """Return the final task-specific convolution for Grad-CAM."""
        if task == "age":
            return self.age_branch[-1].conv2
        if task == "gender":
            return self.gender_branch[-1].conv2
        raise ValueError("task must be 'age' or 'gender'")
