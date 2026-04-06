from pathlib import Path
import shutil
from typing import Dict, List, Set
from sklearn.model_selection import train_test_split

# VAL_SIZE is the proportion of the validation set.
VAL_SIZE = 0.2
RANDOM_STATE = 42
# SPLIT_MODE is used to partition the dataset based on the xtf name or image name.
# Use xtf mode when there are many XTF files, and image mode when there are few.
SPLIT_MODE = "image"  # options: "xtf", "image"

# Root dataset directory
DATASET_ROOT = Path(r"D:\dataset\2025\sept\0803_dataset")

# Source subdirectories
SOURCE_DIRS: Dict[str, Path] = {
    "images": DATASET_ROOT / "images",
    "range": DATASET_ROOT / "range",
    "altitude": DATASET_ROOT / "altitude",
    "shadow": DATASET_ROOT / "shadow",
}

# Output subsets
SUBSETS = ["train", "val"]

def reset_output_dirs(dataset_root: Path, subsets, data_types):
    for subset in subsets:
        subset_root = dataset_root / subset
        if subset_root.exists():
            shutil.rmtree(subset_root)
        for data_type in data_types:
            (subset_root / data_type).mkdir(parents=True, exist_ok=True)

def ensure_directories(directories: List[Path]) -> None:
    """Create directories if they do not already exist."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_prefix(filename: str, split_mode: str = "xtf") -> str:
    """
    Extract grouping prefix from filename.

    Parameters
    ----------
    filename : str
        File name such as 'file1_left_000.png'.
    split_mode : str
        'xtf'   -> group by original xtf file
        'image' -> group by individual image

    Returns
    -------
    str
        Group key.
    """
    stem = Path(filename).stem  # e.g. file1_left_000

    if split_mode == "xtf":
        # Group all segments from the same original xtf file together
        if "_left_" in stem:
            return stem.split("_left_")[0]
        if "_right_" in stem:
            return stem.split("_right_")[0]
        return stem

    elif split_mode == "image":
        # Each image is treated as an independent sample
        return stem

    else:
        raise ValueError("split_mode must be either 'xtf' or 'image'")


def collect_grouped_files(image_dir: Path, split_mode: str = "xtf") -> Dict[str, List[str]]:
    """
    Group image filenames according to the selected split mode.

    Parameters
    ----------
    image_dir : Path
        Directory containing image files.
    split_mode : str
        'xtf'   -> group by original xtf file
        'image' -> group by individual image

    Returns
    -------
    Dict[str, List[str]]
        Mapping from group key to list of image filenames.
    """
    grouped: Dict[str, List[str]] = {}
    image_files = sorted(
        f.name for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    )

    for filename in image_files:
        prefix = get_prefix(filename, split_mode=split_mode)
        grouped.setdefault(prefix, []).append(filename)

    return grouped


def validate_matching_files(grouped_images: Dict[str, List[str]], source_dirs: Dict[str, Path]) -> None:
    """Check whether matching files exist in all source directories.

    For every image file, the function verifies that the corresponding range,
    altitude, and shadow files are present.
    """
    expected_files: Dict[str, Set[str]] = {
        "images": {filename for files in grouped_images.values() for filename in files},
        "range": set(),
        "altitude": set(),
        "shadow": set(),
    }

    for image_filename in expected_files["images"]:
        stem = Path(image_filename).stem
        expected_files["range"].add(f"{stem}.npy")
        expected_files["altitude"].add(f"{stem}.npy")
        expected_files["shadow"].add(f"{stem}.png")

    for key, directory in source_dirs.items():
        actual_files = {f.name for f in directory.iterdir() if f.is_file()}
        missing_files = expected_files[key] - actual_files
        if missing_files:
            missing_preview = sorted(missing_files)[:10]
            raise FileNotFoundError(
                f"Missing {len(missing_files)} file(s) in '{directory}'. "
                f"Examples: {missing_preview}"
            )


def copy_subset_files(
    prefixes: List[str],
    grouped_images: Dict[str, List[str]],
    source_dirs: Dict[str, Path],
    destination_root: Path,
) -> None:
    """Copy all files corresponding to the selected prefixes into one subset."""
    for prefix in prefixes:
        image_filenames = grouped_images[prefix]

        for image_filename in image_filenames:
            stem = Path(image_filename).stem
            paired_filenames = {
                "images": image_filename,
                "range": f"{stem}.npy",
                "altitude": f"{stem}.npy",
                "shadow": f"{stem}.png",
            }

            for key, src_dir in source_dirs.items():
                src_path = src_dir / paired_filenames[key]
                dst_path = destination_root / key / paired_filenames[key]
                shutil.copy2(src_path, dst_path)


def main() -> None:
    """Split the dataset into training and validation subsets."""
    # Create output directories such as train/images, train/range, val/images, etc.

    reset_output_dirs(DATASET_ROOT, SUBSETS, SOURCE_DIRS.keys())
    # Group image files by sample prefix
    grouped_images = collect_grouped_files(SOURCE_DIRS["images"], split_mode=SPLIT_MODE)
    prefixes = sorted(grouped_images.keys())

    if not prefixes:
        raise ValueError(f"No PNG files were found in '{SOURCE_DIRS['images']}'.")

    # Verify that corresponding files exist in all modalities.
    validate_matching_files(grouped_images, SOURCE_DIRS)

    # Split prefixes so that related files stay together.
    train_prefixes, val_prefixes = train_test_split(
        prefixes,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Copy training files.
    copy_subset_files(
        prefixes=train_prefixes,
        grouped_images=grouped_images,
        source_dirs=SOURCE_DIRS,
        destination_root=DATASET_ROOT / "train",
    )

    # Copy validation files.
    copy_subset_files(
        prefixes=val_prefixes,
        grouped_images=grouped_images,
        source_dirs=SOURCE_DIRS,
        destination_root=DATASET_ROOT / "val",
    )

    def count_files_in_dir(directory: Path, suffix: str = None) -> int:
        files = [f for f in directory.iterdir() if f.is_file()]
        if suffix is not None:
            files = [f for f in files if f.suffix.lower() == suffix.lower()]
        return len(files)

    print("Train images:", count_files_in_dir(DATASET_ROOT / "train" / "images", ".png"))
    print("Val images:", count_files_in_dir(DATASET_ROOT / "val" / "images", ".png"))

    print(f"Dataset split completed successfully!")
    print(f"Total groups: {len(prefixes)}")
    print(f"Train groups: {len(train_prefixes)}")
    print(f"Validation groups: {len(val_prefixes)}")


if __name__ == "__main__":
    main()
