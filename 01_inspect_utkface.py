import re
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from age_config import age_to_class


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

METADATA_COLUMNS = [
    "member",
    "age",
    "age_class",
    "gender",
    "race",
    "date",
]
LABEL_COLUMNS = ["age", "gender", "race"]

METADATA_DIR.mkdir(parents=True, exist_ok=True)


def inspect_dataset():
    records = []
    invalid_filenames = []
    corrupted_images = []

    filename_pattern = re.compile(
        r"^(?P<age>\d+)_(?P<gender>[01])_(?P<race>[0-4])_"
        r"(?P<timestamp>\d+)\.jpg\.chip\.jpg$"
    )

    with ZipFile(RAW_ARCHIVE, "r") as archive:
        image_files = sorted(
            member
            for member in archive.namelist()
            if member.lower().endswith(".jpg")
        )

        for member in image_files:
            filename = Path(member).name
            match = filename_pattern.match(filename)

            if match is None:
                invalid_filenames.append(filename)
                continue

            try:
                age = int(match.group("age"))
                gender = int(match.group("gender"))
                race = int(match.group("race"))
                image_bytes = archive.read(member)
                image_hash = sha256(image_bytes).hexdigest()

                with Image.open(BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())

                date = pd.to_datetime(
                    match.group("timestamp"),
                    format="%Y%m%d%H%M%S%f",
                    errors="coerce"
                )

            except (OSError, ValueError, KeyError) as exc:
                corrupted_images.append({
                    "member": member,
                    "reason": str(exc)
                })
                continue

            if pd.isna(date):
                invalid_filenames.append(filename)
                continue

            records.append({
                "member": member,
                "age": age,
                "gender": gender,
                "race": race,
                "date": date,
                "sha256": image_hash,
                "width": width,
                "height": height,
                "channels": channels,
            })

    metadata = pd.DataFrame(records)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"JPG files found   : {len(image_files):,}")
    print(f"Valid images      : {len(metadata):,}")
    print(f"Invalid filenames : {len(invalid_filenames):,}")
    print(f"Corrupted images  : {len(corrupted_images):,}")

    if not metadata.empty:
        print(f"Age range         : {metadata['age'].min()}-{metadata['age'].max()}")
        print("\nGender:")
        print(metadata["gender"].map({0: "Male", 1: "Female"}).value_counts())
        print("\nRace:")
        print(metadata["race"].value_counts().sort_index())

    return metadata


def clean_metadata(metadata):
    before = len(metadata)
    clean = metadata.copy()

    clean = clean.loc[clean["age"].between(0, 116)].copy()
    clean = clean.loc[clean["gender"].isin([0, 1])].copy()
    clean = clean.loc[clean["race"].isin([0, 1, 2, 3, 4])].copy()
    clean = clean.loc[clean["channels"] == 3].copy()

    labels_per_hash = clean.groupby("sha256")[LABEL_COLUMNS].nunique()
    conflict_hashes = labels_per_hash.index[
        labels_per_hash.gt(1).any(axis=1)
    ]
    clean = clean.loc[~clean["sha256"].isin(conflict_hashes)].copy()

    duplicate_files = clean.loc[clean.duplicated("sha256", keep=False)]
    duplicate_group_count = duplicate_files["sha256"].nunique()
    duplicate_file_count = len(duplicate_files)
    clean = clean.drop_duplicates(subset="sha256", keep="first").copy()

    clean["age"] = clean["age"].astype("int64")
    clean["age_class"] = clean["age"].apply(age_to_class).astype("int64")
    clean["gender"] = clean["gender"].astype("int64")
    clean["race"] = clean["race"].astype("int64")
    clean["date"] = pd.to_datetime(clean["date"])

    clean[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "clean_metadata.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    print(f"Before cleaning  : {before:,}")
    print(f"After cleaning   : {len(clean):,}")
    print(f"Removed          : {before - len(clean):,}")
    print(f"Same-label groups: {duplicate_group_count:,}")
    print(f"Same-label files : {duplicate_file_count:,}")
    print(f"Conflict groups  : {len(conflict_hashes):,}")

    return clean


def _make_strata(data):
    return (
        data["age_class"].astype(str)
        + "_"
        + data["gender"].astype(str)
    )


def _safe_strata(data):
    strata = _make_strata(data)
    if strata.value_counts().min() >= 2:
        return strata

    gender_strata = data["gender"].astype(str)
    if gender_strata.value_counts().min() >= 2:
        return gender_strata

    return None


def create_splits(metadata):
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
        raise ValueError("Train/validation/test ratios must sum to 1.")

    strata = _safe_strata(metadata)
    train, temp = train_test_split(
        metadata,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=strata
    )

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    temp_strata = _safe_strata(temp)
    val, test = train_test_split(
        temp,
        test_size=relative_test_ratio,
        stratify=temp_strata
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train[METADATA_COLUMNS].to_csv(METADATA_DIR / "train.csv", index=False)
    val[METADATA_COLUMNS].to_csv(METADATA_DIR / "val.csv", index=False)
    test[METADATA_COLUMNS].to_csv(METADATA_DIR / "test.csv", index=False)

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)
    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(val):,}")
    print(f"Test       : {len(test):,}")

    return train, val, test


def main():
    print("=" * 60)
    print("UTKFACE DATA PREPARATION")
    print("=" * 60)

    metadata = inspect_dataset()
    if metadata.empty:
        raise RuntimeError("No valid UTKFace images were found.")

    metadata = clean_metadata(metadata)
    create_splits(metadata)


if __name__ == "__main__":
    main()
