"""From-scratch Complex CNN for multitask age and gender classification.

This model deliberately does not use transfer learning or torchvision
backbone weights. It keeps the output contract used by the training pipeline:

    {"age": age_logits, "gender": gender_logits}

The age branch fuses intermediate and deep features because age errors in the
current dataset are concentrated around neighboring adult classes.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from age_config import NUM_AGE_CLASSES
except ImportError:
    NUM_AGE_CLASSES = 6


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class ResidualSEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=stride),
            ConvNormAct(out_channels, out_channels),
            SEBlock(out_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
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
            self.skip = nn.Identity()

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        first_stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = [
            ResidualSEBlock(
                in_channels,
                out_channels,
                stride=first_stride,
                dropout=dropout,
            )
        ]
        for _ in range(blocks - 1):
            layers.append(
                ResidualSEBlock(
                    out_channels,
                    out_channels,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class PooledProjection(nn.Module):
    """Project one spatial feature map into a compact vector."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.project(x), 1)


class AgeHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, NUM_AGE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenderHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexCNN(nn.Module):
    """Higher-capacity Complex CNN trained entirely from scratch."""

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvNormAct(3, 32, kernel_size=3),
            ConvNormAct(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
        )

        self.stage1 = ResidualStage(
            64, 64, blocks=2, first_stride=1, dropout=0.02
        )
        self.stage2 = ResidualStage(
            64, 128, blocks=2, first_stride=2, dropout=0.04
        )
        self.stage3 = ResidualStage(
            128, 256, blocks=3, first_stride=2, dropout=0.06
        )
        self.stage4 = ResidualStage(
            256, 512, blocks=2, first_stride=2, dropout=0.08
        )

        # Multi-scale age representation: 128 + 192 + 256 = 576 features.
        self.age_from_stage2 = PooledProjection(128, 128)
        self.age_from_stage3 = PooledProjection(256, 192)
        self.age_from_stage4 = PooledProjection(512, 256)
        self.age_head = AgeHead(576)

        self.gender_head = GenderHead(512)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        feature1 = self.stage1(x)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)

        age_features = torch.cat(
            [
                self.age_from_stage2(feature2),
                self.age_from_stage3(feature3),
                self.age_from_stage4(feature4),
            ],
            dim=1,
        )
        gender_features = torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(feature4, 1), 1
        )

        return {
            "age": self.age_head(age_features),
            "gender": self.gender_head(gender_features),
        }

    def get_gradcam_layer(self, task: str = "age") -> nn.Module:
        if task not in {"age", "gender"}:
            raise ValueError("task must be 'age' or 'gender'")
        return self.stage4.blocks[-1].main[1].block[0]


if __name__ == "__main__":
    model = ComplexCNN()
    sample = torch.randn(2, 3, 224, 224)
    output = model(sample)
    print("age:", tuple(output["age"].shape))
    print("gender:", tuple(output["gender"].shape))
