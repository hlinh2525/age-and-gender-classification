import pandas as pd
import re
from pathlib import Path
from io import BytesIO
from hashlib import sha256
from zipfile import ZipFile
from PIL import Image
from sklearn.model_selection import train_test_split
from age_config import age_to_class


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42

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

            age = int(match.group("age"))
            gender = int(match.group("gender"))
            race = int(match.group("race"))

            try:
                image_bytes = archive.read(member)
                image_hash = sha256(image_bytes).hexdigest()

                with Image.open(BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())

            except (OSError, ValueError) as exc:
                corrupted_images.append(
                    {
                        "member": member,
                        "reason": str(exc)
                    }
                )
                continue

            records.append(
                {
                    "member": member,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "date": pd.to_datetime(
                        match.group("timestamp"),
                        format="%Y%m%d%H%M%S%f"
                    ),
                    "sha256": image_hash,
                    "width": width,
                    "height": height,
                    "channels": channels
                }
            )

    metadata = pd.DataFrame(records)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"JPG files found   : {len(image_files):,}")
    print(f"Valid images      : {len(metadata):,}")
    print(f"Invalid filenames : {len(invalid_filenames):,}")
    if invalid_filenames:
        for filename in invalid_filenames:
            print(f"  - {filename}")

    print(f"Corrupted images  : {len(corrupted_images):,}")

    if not metadata.empty:
        dimensions = metadata[["width", "height"]].value_counts().to_dict()
        print(f"Image dimensions  : {dimensions}")

        print(
            f"Channels          : "
            f"{metadata['channels'].value_counts().sort_index().to_dict()}"
        )

        print(
            f"Age range         : "
            f"{metadata['age'].min()}-{metadata['age'].max()}"
        )

        print("\nGender:")
        print(
            metadata["gender"]
            .map({0: "Male", 1: "Female"})
            .value_counts()
        )

        print("\nRace:")
        print(
            metadata["race"]
            .map({
                0: "White",
                1: "Black",
                2: "Asian",
                3: "Indian",
                4: "Others"
            })
            .value_counts()
        )

    return metadata


def clean_metadata(metadata):
    before = len(metadata)
    clean = metadata.copy()

    clean = clean.loc[clean["age"].between(1, 116)].copy()

    clean = clean.loc[clean["gender"].isin([0, 1])]

    clean = clean.loc[clean["race"].isin([0, 1, 2, 3, 4])]

    clean = clean.loc[clean["channels"] == 3]

    labels_per_hash = clean.groupby("sha256")[LABEL_COLUMNS].nunique()
    conflict_hashes = labels_per_hash.index[
        labels_per_hash.gt(1).any(axis=1)
    ]
    conflict_files = clean.loc[clean["sha256"].isin(conflict_hashes)]
    clean = clean.loc[~clean["sha256"].isin(conflict_hashes)]

    duplicate_files = clean.loc[clean.duplicated("sha256", keep=False)]
    duplicate_group_count = duplicate_files["sha256"].nunique()
    duplicate_file_count = len(duplicate_files)
    removed_duplicate_count = duplicate_file_count - duplicate_group_count
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
    print(f"Duplicate removed: {removed_duplicate_count:,}")
    print(f"Conflict groups  : {len(conflict_hashes):,}")
    print(f"Conflict files   : {len(conflict_files):,}")

    return clean


def create_splits(metadata):
    stratify_labels = metadata.groupby(
        ["age_class", "gender"]
    ).ngroup().astype(str)

    if stratify_labels.value_counts().min() < 2:
        raise ValueError(
            "Stratification groups must contain at least two samples. "
            "Use wider age bins or disable age-gender stratification."
        )

    train, temp = train_test_split(
        metadata,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    temp_stratify_labels = stratify_labels.loc[temp.index]
    if temp_stratify_labels.value_counts().min() < 2:
        temp_stratify_labels = temp["gender"].astype(str)

    val, test = train_test_split(
        temp,
        test_size=relative_test_ratio,
        random_state=RANDOM_STATE,
        stratify=temp_stratify_labels
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    train[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "train.csv",
        index=False
    )

    val[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "val.csv",
        index=False
    )

    test[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "test.csv",
        index=False
    )

    combined = pd.concat(
        [train, val, test],
        ignore_index=True
    )

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(val):,}")
    print(f"Test       : {len(test):,}")

    print("\nGender distribution:")
    print(
        pd.crosstab(
            combined["split"],
            combined["gender"],
            normalize="index"
        ).rename(
            columns={
                0: "Male",
                1: "Female"
            }
        )
    )

    return train, val, test


def main():
    print("=" * 60)
    print("UTKFACE DATA PREPARATION")
    print("=" * 60)

    metadata = inspect_dataset()
    if metadata.empty:
        raise RuntimeError(
            "No valid UTKFace images were found."
        )

    metadata = clean_metadata(metadata)

    train, val, test = create_splits(metadata)

if __name__ == "__main__":
    main()