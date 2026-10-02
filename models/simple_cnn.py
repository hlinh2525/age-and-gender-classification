import torch
from torch import nn

from age_config import NUM_AGE_CLASSES


class SimpleCNN(nn.Module):
    """Three-block Simple CNN for age and gender classification."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.age_head = nn.Linear(128, NUM_AGE_CLASSES)
        self.gender_head = nn.Linear(128, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

    def forward(self, images):
        features = self.features(images)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        features = self.shared_fc(features)

        return {
            "age": self.age_head(features),
            "gender": self.gender_head(features),
        }

    def get_gradcam_layer(self):
        # features[6] is Conv2d(64, 128, ...), the final convolution layer.
        return self.features[6]
