"""
Getting NABirds onto disk, auditing every file, and building the partitions.

The Activeloop copy stores each photograph as its original JPEG, so the export
writes those bytes unchanged instead of decoding and re-encoding them, which
would add a second round of compression artifacts to every image. Labels index
into the full 1,011-node NABirds taxonomy; only the 555 leaf categories are
ever used as labels, and they are re-indexed to 0..554 for the models.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from bird_species.config import (
    ACTIVELOOP_SPLITS,
    CLASSES_PATH,
    EXPORT_WORKERS,
    HASH_SIZE,
    IMAGE_SIZE,
    MAX_SHOTS,
    METADATA_PATH,
    MIN_IMAGE_SIDE,
    MOMENT_SIZE,
    N_JOBS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    RESIZE_SIZE,
    VALIDATION_FRACTION,
)

VARIANT_PATTERN = re.compile(r"^(?P<species>.*?)\s*\((?P<variant>.*)\)$")
BOX_COLUMNS = ["box_x", "box_y", "box_w", "box_h"]


def split_class_name(name: str) -> tuple[str, str]:
    """
    Split a class name into species and variant.

    `'Common Eider (Adult male)'` becomes `('Common Eider', 'Adult male')`; a plain name has no
    variant.
    """
    match = VARIANT_PATTERN.match(name)
    if match is None:
        return name, ""
    return match["species"], match["variant"]


def class_table(names: Sequence[str], used_labels: Sequence[int]) -> pd.DataFrame:
    """The leaf categories that appear as labels, re-indexed to 0..n-1 in taxonomy order."""
    used = sorted(int(label) for label in set(used_labels))
    species, variants = zip(*(split_class_name(names[label]) for label in used), strict=True)
    return pd.DataFrame(
        {
            "label": used,
            "class_index": range(len(used)),
            "class_name": [names[label] for label in used],
            "species": species,
            "variant": variants,
        }
    )


def export_range(path: str, split: str, start: int, stop: int, raw_dir: Path) -> int:
    """
    Write samples `start..stop-1` of one split. Files already on disk are skipped, so a rerun
    resumes.

    Each file is written under a temporary name and renamed when complete, so
    an interrupted run never leaves a truncated JPEG that a rerun would keep.
    """
    import deeplake

    images = deeplake.load(path, read_only=True, verbose=False)["images"]
    written = 0
    for index in range(start, stop):
        target = raw_dir / f"{split}/{index:05d}.jpg"
        if not target.exists():
            partial = target.with_suffix(".part")
            partial.write_bytes(images[index].tobytes())
            partial.replace(target)
            written += 1
    return written


def export_split(
    split: str, path: str, raw_dir: Path, workers: int = EXPORT_WORKERS
) -> tuple[pd.DataFrame, list[str]]:
    """
    Write one Activeloop split's JPEGs to `raw_dir/split` and return its metadata rows.

    The download is bound by network round trips, not by the CPU, so several
    processes each stream their own range of samples.
    """
    # Imported here, not at the top: only a first run downloads, and the deeplake
    # 3.x client the Activeloop copy requires warns about NumPy internals on import.
    import deeplake

    dataset = deeplake.load(path, read_only=True, verbose=False)
    names = list(dataset["labels"].info["class_names"])
    labels = dataset["labels"].numpy(aslist=False).ravel()
    boxes = dataset["boxes"].numpy(aslist=True)
    shapes = dataset["images"].shapes()
    (raw_dir / split).mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, len(dataset), workers + 1).astype(int)
    Parallel(n_jobs=workers)(
        delayed(export_range)(path, split, int(start), int(stop), raw_dir)
        for start, stop in zip(edges[:-1], edges[1:], strict=True)
    )
    first_box = np.array([np.asarray(box, dtype=float).reshape(-1, 4)[0] for box in boxes])
    rows = pd.DataFrame(
        {
            "split": split,
            "index": np.arange(len(dataset)),
            "filename": [f"{split}/{index:05d}.jpg" for index in range(len(dataset))],
            "label": labels.astype(int),
            "box_x": first_box[:, 0],
            "box_y": first_box[:, 1],
            "box_w": first_box[:, 2],
            "box_h": first_box[:, 3],
            "height": shapes[:, 0],
            "width": shapes[:, 1],
        }
    )
    return rows, names


def download_nabirds(
    raw_dir: Path = RAW_DIR, splits: dict[str, str] = ACTIVELOOP_SPLITS
) -> pd.DataFrame:
    """
    Export NABirds once and return the metadata table; later calls read the cache.

    Boxes are stored as left, top, width, height in pixels, the format of the
    official `bounding_boxes.txt`.
    """
    metadata_path, classes_path = raw_dir / METADATA_PATH.name, raw_dir / CLASSES_PATH.name
    if metadata_path.exists() and classes_path.exists():
        return pd.read_csv(metadata_path)
    parts, names = [], None
    for split, path in splits.items():
        rows, names = export_split(split, path, raw_dir)
        parts.append(rows)
    metadata = pd.concat(parts, ignore_index=True)
    classes = class_table(names, metadata["label"])
    metadata = metadata.merge(classes[["label", "class_index"]], on="label", how="left")
    metadata.to_csv(metadata_path, index=False)
    classes.to_csv(classes_path, index=False)
    return metadata


def load_classes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / CLASSES_PATH.name, keep_default_na=False)


def load_image(filename: str, raw_dir: Path = RAW_DIR) -> Image.Image:
    """An RGB copy of one photograph, whatever mode the file was saved in."""
    with Image.open(raw_dir / filename) as image:
        return image.convert("RGB")


def audit_file(filename: str, raw_dir: Path = RAW_DIR, hash_size: int = HASH_SIZE) -> dict:
    """
    Decode one file completely and fingerprint it.

    `verify` only checks the container, so the image is reopened and fully
    decoded: a truncated JPEG passes `verify` and fails on `load`. The SHA-256
    of the bytes finds identical files; the perceptual hash is kept to test
    whether layout alone can find the subtler copies.
    """
    path = raw_dir / filename
    data = path.read_bytes()
    row = {
        "filename": filename,
        "file_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "decodes": True,
        "mode": "",
        "decoded_width": 0,
        "decoded_height": 0,
        "phash": "",
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            row.update(
                mode=image.mode,
                decoded_width=image.width,
                decoded_height=image.height,
                phash=str(imagehash.phash(image.convert("RGB"), hash_size=hash_size)),
            )
    except (OSError, SyntaxError, ValueError):
        row["decodes"] = False
    return row


def audit_images(
    metadata: pd.DataFrame, raw_dir: Path = RAW_DIR, n_jobs: int = N_JOBS
) -> pd.DataFrame:
    """The audit row of every file, joined back to its metadata, with the box checks."""
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(audit_file)(name, raw_dir) for name in metadata["filename"]
    )
    audit = metadata.merge(pd.DataFrame(rows), on="filename")
    audit["shape_matches"] = (audit["decoded_width"] == audit["width"]) & (
        audit["decoded_height"] == audit["height"]
    )
    audit["too_small"] = audit[["decoded_width", "decoded_height"]].min(axis=1) < MIN_IMAGE_SIDE
    audit["box_overshoot"] = np.maximum(
        audit["box_x"] + audit["box_w"] - audit["width"],
        audit["box_y"] + audit["box_h"] - audit["height"],
    ).clip(lower=0)
    audit["box_inside"] = (
        (audit["box_x"] >= 0) & (audit["box_y"] >= 0) & (audit["box_overshoot"] <= 1)
    )
    return audit


def audit_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """One row per split: how many files fail each check."""
    checks = audit.assign(
        fails_decode=~audit["decodes"],
        not_rgb=audit["mode"] != "RGB",
        shape_mismatch=~audit["shape_matches"],
        box_outside=~audit["box_inside"],
        identical_bytes=audit.duplicated("sha256", keep="first"),
    )
    columns = [
        "fails_decode",
        "not_rgb",
        "shape_mismatch",
        "too_small",
        "box_outside",
        "identical_bytes",
    ]
    summary = checks.groupby("split")[columns].sum()
    summary.insert(0, "images", checks.groupby("split").size())
    return summary.reset_index()


def clip_boxes(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Boxes cut back to the photograph's edges, and the bird's share of the frame.

    Some annotations run past the image, typically when a bird fills the frame
    and the box was drawn to the full image width from an offset start. The
    bird does reach the edge in those photographs, so the box is clipped
    rather than the photograph dropped.
    """
    right = np.minimum(frame["box_x"] + frame["box_w"], frame["width"])
    bottom = np.minimum(frame["box_y"] + frame["box_h"], frame["height"])
    clipped = frame.assign(box_w=right - frame["box_x"], box_h=bottom - frame["box_y"])
    clipped["box_fraction"] = (
        clipped["box_w"] * clipped["box_h"] / (clipped["width"] * clipped["height"])
    )
    clipped["box_center_x"] = (clipped["box_x"] + clipped["box_w"] / 2) / clipped["width"]
    clipped["box_center_y"] = (clipped["box_y"] + clipped["box_h"] / 2) / clipped["height"]
    return clipped


def crop_geometry(
    width: float, height: float, resize: int = RESIZE_SIZE, crop: int = IMAGE_SIZE
) -> tuple[float, float, float]:
    """The scale and offsets that map original pixels into the resized, center-cropped input."""
    scale = resize / min(width, height)
    return scale, (width * scale - crop) / 2, (height * scale - crop) / 2


def box_in_crop(
    box: Sequence[float],
    width: float,
    height: float,
    resize: int = RESIZE_SIZE,
    crop: int = IMAGE_SIZE,
) -> tuple[float, float, float, float]:
    """A left-top-width-height box as corners inside the model's input crop, clipped to it."""
    scale, left, top = crop_geometry(width, height, resize, crop)
    x0, y0 = box[0] * scale - left, box[1] * scale - top
    x1, y1 = x0 + box[2] * scale, y0 + box[3] * scale
    return (
        float(np.clip(x0, 0, crop)),
        float(np.clip(y0, 0, crop)),
        float(np.clip(x1, 0, crop)),
        float(np.clip(y1, 0, crop)),
    )


def box_kept_by_crop(
    frame: pd.DataFrame, resize: int = RESIZE_SIZE, crop: int = IMAGE_SIZE
) -> np.ndarray:
    """The share of each bird's box that survives the center crop every model sees."""
    kept = []
    for row in frame[["width", "height", *BOX_COLUMNS]].itertuples(index=False):
        scale = resize / min(row.width, row.height)
        x0, y0, x1, y1 = box_in_crop(row[2:], row.width, row.height, resize, crop)
        kept.append((x1 - x0) * (y1 - y0) / max(row.box_w * row.box_h * scale**2, 1e-9))
    return np.asarray(kept)


def carve_validation(
    pool: pd.DataFrame,
    fraction: float = VALIDATION_FRACTION,
    shots: int = MAX_SHOTS,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """
    Mark a stratified, group-aware validation share of the training pool.

    Whole duplicate groups move together, so no photograph has a copy on the
    other side of the boundary. A class never gives up so many groups that
    fewer than `shots` remain for a few-shot start.
    """
    rng = np.random.default_rng(seed)
    in_validation = pd.Series(False, index=pool.index)
    for _, members in pool.groupby("class_index"):
        groups = members["group"].unique()
        rng.shuffle(groups)
        take = min(int(round(fraction * len(groups))), max(len(groups) - shots, 0))
        chosen = set(groups[:take])
        in_validation[members.index[members["group"].isin(chosen)]] = True
    return in_validation


def assign_partitions(clean: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Label each kept photograph train, validation or test; test is the cleaned test split."""
    data = clean.copy()
    data["partition"] = np.where(data["split"] == "test", "test", "train")
    pool = data[data["partition"] == "train"]
    data.loc[pool.index[carve_validation(pool, seed=seed)], "partition"] = "validation"
    return data


def partition_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Images, classes and per-class counts for each partition."""
    counts = data.groupby(["partition", "class_index"]).size()
    summary = counts.groupby("partition").agg(
        images="sum",
        classes="size",
        per_class_min="min",
        per_class_median="median",
        per_class_max="max",
    )
    return summary.reindex(["train", "validation", "test"]).reset_index()


def draw_shots(pool: pd.DataFrame, shots: int, seed: int = RANDOM_SEED) -> pd.Index:
    """At most `shots` photographs per class, one per duplicate group, drawn with `seed`."""
    rng = np.random.default_rng(seed)
    chosen = []
    for _, members in pool.groupby("class_index"):
        one_per_group = members.drop_duplicates("group")
        picks = rng.choice(len(one_per_group), size=min(shots, len(one_per_group)), replace=False)
        chosen.extend(one_per_group.index[np.sort(picks)])
    return pd.Index(chosen)


def _moment_chunk(
    filenames: Sequence[str], size: int, raw_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    resize = int(round(size * RESIZE_SIZE / IMAGE_SIZE))
    total = np.zeros((size, size, 3))
    squares = np.zeros((size, size, 3))
    for name in filenames:
        image = load_image(name, raw_dir)
        scale = resize / min(image.size)
        image = image.resize(
            (max(size, round(image.width * scale)), max(size, round(image.height * scale))),
            Image.BICUBIC,
        )
        left, top = (image.width - size) // 2, (image.height - size) // 2
        pixels = (
            np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.float64) / 255.0
        )
        total += pixels
        squares += pixels**2
    return total, squares


def pixel_moments(
    filenames: Sequence[str], size: int = MOMENT_SIZE, raw_dir: Path = RAW_DIR, n_jobs: int = N_JOBS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel-wise mean and standard deviation over photographs framed the way the models frame them.

    Sums and sums of squares are accumulated in parallel chunks, so no stack
    of images is ever held in memory.
    """
    chunks = np.array_split(np.asarray(filenames), max(1, abs(n_jobs) * 4))
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_moment_chunk)(chunk, size, raw_dir) for chunk in chunks
    )
    total = sum(part[0] for part in parts)
    squares = sum(part[1] for part in parts)
    mean = total / len(filenames)
    return mean, np.sqrt(np.maximum(squares / len(filenames) - mean**2, 0))


def variant_examples(
    frame: pd.DataFrame,
    classes: pd.DataFrame,
    species_count: int,
    per_species: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Example photographs of species that NABirds splits into several classes.

    The first `species_count` species with at least `per_species` classes each contribute one
    photograph of each of their first `per_species` classes.
    """
    variants = classes.groupby("species", sort=False)["class_index"].apply(list)
    chosen = variants[variants.map(len) >= per_species].head(species_count)
    rows = [
        frame[frame["class_index"] == class_index].sample(1, random_state=seed)
        for class_indices in chosen
        for class_index in class_indices[:per_species]
    ]
    return pd.concat(rows).merge(
        classes[["class_index", "class_name"]], on="class_index", how="left"
    )


def write_processed(tables: dict[str, pd.DataFrame], directory: Path = PROCESSED_DIR) -> list[str]:
    """Write each table as `<name>.csv` and return the file names."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    return sorted(f"{name}.csv" for name in tables)
