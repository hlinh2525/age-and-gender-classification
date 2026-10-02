from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import json
import hashlib
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from age_config import AGE_LABELS, NUM_AGE_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessing"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2

SPLIT_NAMES = ("train", "val", "test")
COLUMNS = ["member", "age_class", "gender", "race"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_splits(metadata_dir=METADATA_DIR):
    metadata_dir = Path(metadata_dir)

    return {
        split: pd.read_csv(
            metadata_dir / f"{split}.csv",
            usecols=COLUMNS,
            dtype={
                "member": "string",
                "age_class": "int64",
                "gender": "int64",
                "race": "int64",
            },
            encoding="utf-8-sig",
        )
        for split in SPLIT_NAMES
    }


def create_zip_index(zip_path, members):
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"UTKFace ZIP not found: {zip_path}")

    requested = {
        str(member).strip().replace("\\", "/")
        for member in members
    }

    with ZipFile(zip_path) as archive:
        archive_members = set(archive.namelist())

    basename_map = {}
    for member in archive_members:
        basename_map.setdefault(Path(member).name, []).append(member)

    index = {}
    missing = []

    for member in requested:
        if member in archive_members:
            index[member] = member
            continue

        candidates = basename_map.get(Path(member).name, [])
        if len(candidates) == 1:
            index[member] = candidates[0]
        else:
            missing.append(member)

    if missing:
        raise FileNotFoundError(
            f"ZIP is missing {len(missing)} images. "
            f"Examples: {sorted(missing)[:3]}"
        )

    return index


def create_transform(model_type, training, image_size=IMAGE_SIZE):
    if model_type not in {"scratch", "resnet18"}:
        raise ValueError("model_type must be scratch or resnet18.")

    steps = [transforms.Resize((image_size, image_size))]

    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
        ])

    steps.append(transforms.ToTensor())

    if model_type == "resnet18":
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(steps)


class UTKFaceDataset(Dataset):
    def __init__(self, data, zip_path, index, transform):
        self.data = data.reset_index(drop=True)
        self.zip_path = Path(zip_path)
        self.index = index
        self.transform = transform
        self.archive = None

    def __len__(self):
        return len(self.data)

    def _get_archive(self):
        if self.archive is None:
            self.archive = ZipFile(self.zip_path)
        return self.archive

    def __getitem__(self, position):
        row = self.data.iloc[position]
        member = str(row["member"])
        content = self._get_archive().read(self.index[member])

        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "age_class": torch.tensor(
                int(row["age_class"]),
                dtype=torch.long,
            ),
            "gender": torch.tensor(
                int(row["gender"]),
                dtype=torch.long,
            ),
            "member": member,
        }

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["archive"] = None
        return state

    def __del__(self):
        if hasattr(self, "archive"):
            self.close()


def build_loaders(
    model_type="scratch",
    metadata_dir=METADATA_DIR,
    zip_path=ZIP_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    if image_size < 32 or batch_size < 1:
        raise ValueError("image_size must be at least 32 and batch_size at least 1.")

    splits = load_splits(metadata_dir)
    members = pd.concat(splits.values())["member"]
    index = create_zip_index(zip_path, members)

    loaders = {}
    for split, data in splits.items():
        dataset = UTKFaceDataset(
            data=data,
            zip_path=zip_path,
            index=index,
            transform=create_transform(
                model_type=model_type,
                training=(split == "train"),
                image_size=image_size,
            ),
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def check_batches(loaders, model_type, image_size=IMAGE_SIZE):
    for split, loader in loaders.items():
        batch = next(iter(loader))
        size = len(batch["member"])

        assert batch["image"].shape == (size, 3, image_size, image_size)
        assert batch["age_class"].shape == (size,)
        assert batch["gender"].shape == (size,)
        assert batch["age_class"].ge(0).all()
        assert batch["age_class"].lt(NUM_AGE_CLASSES).all()
        assert ((batch["gender"] == 0) | (batch["gender"] == 1)).all()

        if model_type == "scratch":
            assert batch["image"].min() >= 0
            assert batch["image"].max() <= 1

        print(model_type, split, tuple(batch["image"].shape))


def save_config(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "color": "RGB",
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "gender_encoding": {"0": "Male", "1": "Female"},
        "age_target": "age_class",
        "age_labels": AGE_LABELS,
        "age_num_classes": NUM_AGE_CLASSES,
        "scratch": {"pixel_range": [0, 1]},
        "resnet18": {
            "pixel_range_before_normalize": [0, 1],
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "train_augmentation": {
            "horizontal_flip_probability": 0.5,
            "brightness": 0.1,
            "contrast": 0.1,
        },
        "validation_test_augmentation": False,
        "split_sha256": {
            split: hashlib.sha256(
                (METADATA_DIR / f"{split}.csv").read_bytes()
            ).hexdigest()
            for split in SPLIT_NAMES
        },
    }

    (output_dir / "preprocessing_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_config(OUTPUT_DIR)
