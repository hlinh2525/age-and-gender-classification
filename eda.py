from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from age_config import AGE_LABELS


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

EDA_DIR = PROJECT_ROOT / "data" / "eda"
FIGURES_DIR = EDA_DIR / "figures"

RANDOM_STATE = 42
IMAGE_SAMPLE_SIZE = 500      

GENDER_MAP = {0: "Male", 1: "Female"}
RACE_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

sns.set_theme(style="white")
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_metadata():
    clean = pd.read_csv(METADATA_DIR / "clean_metadata.csv")
    splits = {}
    for name in ("train", "val", "test"):
        path = METADATA_DIR / f"{name}.csv"
        if path.exists():
            splits[name] = pd.read_csv(path)

    print("=" * 60)
    print("LOAD DATA")
    print("=" * 60)
    print(f"clean_metadata.csv : {len(clean):,} rows")
    for name, df in splits.items():
        print(f"{name}.csv".ljust(20) + f": {len(df):,} rows")

    return clean, splits


# ============================================================
# 2. OVERVIEW & DATA QUALITY
# ============================================================

def overview(metadata):
    print("\n" + "=" * 60)
    print("OVERVIEW & DATA QUALITY")
    print("=" * 60)

    print(f"Shape             : {metadata.shape}")
    print(f"Dtypes            :\n{metadata.dtypes}")
    print(f"Missing values    :\n{metadata.isna().sum()}")
    print(f"Duplicate rows    : {metadata.duplicated().sum()}")
    print(f"Duplicate fnames  : {metadata['member'].duplicated().sum()}")

    print("\nDescriptive statistics (age):")
    print(metadata["age"].describe())

    return {
        "shape": metadata.shape,
        "missing": metadata.isna().sum().to_dict(),
        "duplicates": int(metadata.duplicated().sum()),
    }


# ============================================================
# 3. UNIVARIATE ANALYSIS
# ============================================================

def univariate_age(metadata):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(metadata["age"], bins=40, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Age distribution (histogram)")
    axes[0].set_xlabel("Age")

    sns.boxplot(x=metadata["age"], ax=axes[1], color="#DD8452")
    axes[1].set_title("Age distribution (boxplot)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_age_distribution.png", dpi=150)
    plt.close()

    q1, q3 = metadata["age"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = metadata[(metadata["age"] < lower) | (metadata["age"] > upper)]

    print("\n" + "=" * 60)
    print("UNIVARIATE: AGE")
    print("=" * 60)
    print(f"Mean   : {metadata['age'].mean():.2f}")
    print(f"Median : {metadata['age'].median():.2f}")
    print(f"Std    : {metadata['age'].std():.2f}")
    print(f"Skew   : {metadata['age'].skew():.2f}")
    print(f"IQR outlier bounds : [{lower:.1f}, {upper:.1f}]")
    print(f"IQR outlier count  : {len(outliers):,} ({len(outliers)/len(metadata)*100:.2f}%)")

def age_class_table(metadata):
    counts = (
        metadata["age_class"]
        .value_counts()
        .reindex(range(len(AGE_LABELS)), fill_value=0)
    )
    percentages = counts / counts.sum() * 100

    table = pd.DataFrame({
        "count": counts,
        "pct": percentages.round(2),
    })
    table.index = AGE_LABELS

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        x=table.index,
        y=table["count"].values,
        color="#4C72B0"
    )
    plt.title("Sample count per age class")
    plt.xlabel("Age class")
    plt.ylabel("Count")
    for index, (count, percentage) in enumerate(
        zip(table["count"], table["pct"])
    ):
        ax.text(
            index,
            count,
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=8
        )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_age_class_distribution.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("AGE CLASS DISTRIBUTION")
    print("=" * 60)
    print(table)
    return table


def univariate_gender(metadata):
    counts = metadata["gender"].map(GENDER_MAP).value_counts()

    plt.figure(figsize=(6, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="pastel")
    plt.title("Gender distribution")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}\n({v/counts.sum()*100:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_gender_distribution.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("UNIVARIATE: GENDER")
    print("=" * 60)
    print(counts)
    print(f"Imbalance ratio (max/min): {counts.max()/counts.min():.2f}")


def univariate_race(metadata):
    counts = metadata["race"].map(RACE_MAP).value_counts()

    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="muted", order=counts.index)
    plt.title("Race distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}\n({v/counts.sum()*100:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_race_distribution.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("UNIVARIATE: RACE")
    print("=" * 60)
    print(counts)
    print(f"Imbalance ratio (max/min): {counts.max()/counts.min():.2f}")


# ============================================================
# 4. BIVARIATE ANALYSIS
# ============================================================

def bivariate_age_gender(metadata):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=metadata,
        x="age",
        hue=metadata["gender"].map(GENDER_MAP),
        fill=True,
        common_norm=False,
        alpha=0.4,
    )
    plt.title("Age distribution by gender")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_age_by_gender.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("BIVARIATE: AGE x GENDER")
    print("=" * 60)
    print(metadata.groupby(metadata["gender"].map(GENDER_MAP))["age"].describe())


def bivariate_age_race(metadata):
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x=metadata["race"].map(RACE_MAP),
        y=metadata["age"],
        palette="muted",
    )
    plt.title("Age distribution by race")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_age_by_race.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("BIVARIATE: AGE x RACE")
    print("=" * 60)
    print(metadata.groupby(metadata["race"].map(RACE_MAP))["age"].describe())


def bivariate_gender_race(metadata):
    cross = pd.crosstab(
        metadata["race"].map(RACE_MAP),
        metadata["gender"].map(GENDER_MAP),
        normalize="index",
    )

    plt.figure(figsize=(8, 5))
    cross.plot(kind="bar", stacked=True, colormap="Set2", ax=plt.gca())
    plt.title("Gender ratio within each race group")
    plt.ylabel("Proportion")
    plt.xticks(rotation=20)
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_gender_by_race.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("BIVARIATE: GENDER x RACE (row-normalized)")
    print("=" * 60)
    print(cross)


# ============================================================
# 5. IMAGE PROPERTY ANALYSIS (sampled directly from the zip)
# ============================================================

def image_property_analysis(metadata, sample_size=IMAGE_SAMPLE_SIZE):
    sample_member = (
        metadata["member"]
        .sample(n=min(sample_size, len(metadata)), random_state=RANDOM_STATE)
        .tolist()
    )

    records = []
    with ZipFile(RAW_ARCHIVE, "r") as archive:
        name_lookup = {Path(n).name: n for n in archive.namelist()}

        for member in sample_member:
            filename = Path(member).name
            if filename not in name_lookup:
                continue

            raw_bytes = archive.read(member)
            with Image.open(BytesIO(raw_bytes)) as img:
                img.load()
                width, height = img.size
                channels = len(img.getbands())
                arr = np.asarray(img.convert("L"), dtype=np.float32)
                brightness = arr.mean()
                contrast = arr.std()

            records.append(
                {
                    "member": member,
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "file_size_kb": len(raw_bytes) / 1024,
                    "brightness": brightness,
                    "contrast": contrast,
                }
            )

    props = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.histplot(props["file_size_kb"], bins=30, ax=axes[0], color="#55A868")
    axes[0].set_title("File size (KB)")

    sns.histplot(props["brightness"], bins=30, ax=axes[1], color="#C44E52")
    axes[1].set_title("Mean brightness (0-255)")

    sns.histplot(props["contrast"], bins=30, ax=axes[2], color="#8172B2")
    axes[2].set_title("Contrast (std of pixel intensity)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_image_properties.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print(f"IMAGE PROPERTIES (sample n={len(props):,})")
    print("=" * 60)
    print(f"Dimensions found  : {props[['width', 'height']].drop_duplicates().values.tolist()}")
    print(f"Channels          : {props['channels'].value_counts().to_dict()}")
    print(props[["file_size_kb", "brightness", "contrast"]].describe())

    return props

def image_quality_vs_age(props, metadata):
    merged = props.merge(
        metadata[["member", "age"]], on="member", how="inner")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.regplot(
        data=merged, x="age", y="brightness",
        scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "red"},
        ax=axes[0], )
    axes[0].set_title("Brightness vs Age")

    sns.regplot(
        data=merged, x="age", y="contrast",
        scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "red"},
        ax=axes[1],
    )
    axes[1].set_title("Contrast vs Age")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_image_quality_vs_age.png", dpi=150)
    plt.close()

    corr_brightness = merged["age"].corr(merged["brightness"])
    corr_contrast = merged["age"].corr(merged["contrast"])
    corr_filesize = merged["age"].corr(merged["file_size_kb"])

    print("\n" + "=" * 60)
    print("IMAGE QUALITY vs AGE")
    print("=" * 60)
    print(f"Correlation age-brightness : {corr_brightness:.3f}")
    print(f"Correlation age-contrast   : {corr_contrast:.3f}")
    print(f"Correlation age-file_size  : {corr_filesize:.3f}")
    print(
        "Rule of thumb: |corr| < 0.1 negligible, 0.1-0.3 weak/worth noting, "
        "> 0.3 systematic bias likely present."
    )

    return {
        "corr_brightness": corr_brightness,
        "corr_contrast": corr_contrast,
        "corr_filesize": corr_filesize,
    }


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(metadata):
    encoded = metadata[["age", "gender"]].copy()
    race_dummies = pd.get_dummies(metadata["race"].map(RACE_MAP), prefix="race")
    encoded = pd.concat([encoded, race_dummies], axis=1)

    corr = encoded.corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix (age, gender, race one-hot)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_correlation_matrix.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    print(corr["age"].sort_values(ascending=False))

    return corr


# ============================================================
# 7. SPLIT DISTRIBUTION COMPARISON
# ============================================================

def split_distribution_comparison(splits):
    if not splits:
        print("\nNo split files found, skipping split comparison.")
        return

    combined = pd.concat(
        [df.assign(split=name) for name, df in splits.items()],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(
        data=combined, x="age", hue="split", common_norm=False, ax=axes[0]
    )
    axes[0].set_title("Age distribution across splits")

    gender_pct = (
        pd.crosstab(combined["split"], combined["gender"], normalize="index") * 100
    ).rename(columns=GENDER_MAP)
    gender_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="Set2")
    axes[1].set_title("Gender proportion (%) across splits")
    axes[1].set_ylabel("%")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_split_comparison.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("SPLIT DISTRIBUTION COMPARISON")
    print("=" * 60)
    for name, df in splits.items():
        print(f"\n{name.upper()} (n={len(df):,})")
        print(f"  Age  : mean={df['age'].mean():.2f}, std={df['age'].std():.2f}")
        print(f"  Gender %: {(df['gender'].value_counts(normalize=True)*100).round(2).to_dict()}")
        print(f"  Race %  : {(df['race'].value_counts(normalize=True)*100).round(2).to_dict()}")


# ============================================================
# 8. SUMMARY REPORT
# ============================================================

def generate_summary_report(metadata, overview_stats, props, corr):
    lines = []
    lines.append("UTKFace EDA Summary Report")
    lines.append("=" * 60)
    lines.append(f"Total cleaned samples : {overview_stats['shape'][0]:,}")
    lines.append(f"Duplicate rows        : {overview_stats['duplicates']}")
    lines.append("")
    lines.append(f"Age   : min={metadata['age'].min()}, max={metadata['age'].max()}, "
                  f"mean={metadata['age'].mean():.2f}, std={metadata['age'].std():.2f}")
    lines.append(f"Gender ratio (M:F)    : "
                  f"{(metadata['gender'] == 0).sum():,} : {(metadata['gender'] == 1).sum():,}")
    lines.append(f"Race distribution     : "
                  f"{metadata['race'].map(RACE_MAP).value_counts().to_dict()}")
    lines.append("")
    lines.append(f"Sampled image size(s) : "
                  f"{props[['width','height']].drop_duplicates().values.tolist()}")
    lines.append(f"Mean brightness       : {props['brightness'].mean():.2f}")
    lines.append(f"Mean file size (KB)   : {props['file_size_kb'].mean():.2f}")
    lines.append("")
    lines.append("Correlation with age:")
    for k, v in corr["age"].sort_values(ascending=False).items():
        if k != "age":
            lines.append(f"  {k}: {v:.3f}")
    lines.append("")
    lines.append("Figures saved in data/eda/figures/:")
    for fig_path in sorted(FIGURES_DIR.glob("*.png")):
        lines.append(f"  - {fig_path.name}")

    report_path = EDA_DIR / "summary_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    print(f"Saved to {report_path}")


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("UTKFACE EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    clean, splits = load_metadata()
    overview_stats = overview(clean)

    univariate_age(clean)
    age_class_table(clean)
    univariate_gender(clean)
    univariate_race(clean)

    bivariate_age_gender(clean)
    bivariate_age_race(clean)
    bivariate_gender_race(clean)

    props = image_property_analysis(clean)
    image_quality_vs_age(props, clean)

    corr = correlation_analysis(clean)
    split_distribution_comparison(splits)

    generate_summary_report(clean, overview_stats, props, corr)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"Figures -> {FIGURES_DIR}")
    print(f"Report  -> {EDA_DIR / 'summary_report.txt'}")


if __name__ == "__main__":
    main()