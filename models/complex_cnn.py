from torch import nn

from age_config import NUM_AGE_CLASSES


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def make_stage(in_channels, out_channels, num_blocks, stride):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_channels, out_channels, 1))
    return nn.Sequential(*layers)


class ComplexCNN(nn.Module):
    """
    Nâng cấp từ SimpleCNN: mạng CNN sâu hơn với các residual block (giống
    tinh thần ResNet) nhưng khởi tạo và huấn luyện HOÀN TOÀN TỪ ĐẦU (không
    dùng trọng số pretrained ImageNet).

    Kiến trúc:
    - Stem + 3 stage residual dùng CHUNG cho cả 2 nhiệm vụ (shared trunk)
    - Sau đó TÁCH NHÁNH: age_branch và gender_branch, mỗi nhánh có 1 stage
      residual độc lập, không chia sẻ trọng số với nhau
    - Head riêng cho age classification và gender classification

    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # Stem: giảm kích thước ảnh đầu vào (224 -> 56)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Trunk dùng chung
        self.shared_stage1 = make_stage(32, 64, num_blocks=2, stride=1)
        self.shared_stage2 = make_stage(64, 128, num_blocks=2, stride=2)
        self.shared_stage3 = make_stage(128, 256, num_blocks=2, stride=2)

        # Hai nhánh độc lập, khởi tạo ngẫu nhiên riêng biệt
        self.age_branch = make_stage(256, 512, num_blocks=2, stride=2)
        self.gender_branch = make_stage(256, 512, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.age_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_AGE_CLASSES)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, images):
        x = self.stem(images)
        x = self.shared_stage1(x)
        x = self.shared_stage2(x)
        shared = self.shared_stage3(x)

        age_features = self.pool(self.age_branch(shared)).flatten(1)
        gender_features = self.pool(self.gender_branch(shared)).flatten(1)

        return {
            "age": self.age_head(age_features),          # (B, 6), age-class logits
            "gender": self.gender_head(gender_features)  # (B, 2), logits
        }

    def get_gradcam_layer(self, task="age"):
        if task == "age":
            return self.age_branch[-1].conv2
        if task == "gender":
            return self.gender_branch[-1].conv2
        raise ValueError("task must be age or gender")