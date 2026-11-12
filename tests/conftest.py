"""
Shared fixtures: small synthetic photographs written to a temporary folder.

The suite never touches NABirds. A synthetic photograph is a flat background
with a colored rectangle standing in for the bird, so a test can place the
"bird" at a known position and check what the code says about it.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")


def make_photo(
    width: int = 320,
    height: int = 240,
    color=(30, 120, 200),
    box=(100, 60, 80, 60),
    box_color=(220, 40, 40),
) -> Image.Image:
    """A flat-colored photograph with one rectangle (left, top, width, height) painted on it."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:] = color
    left, top, box_width, box_height = box
    pixels[top : top + box_height, left : left + box_width] = box_color
    return Image.fromarray(pixels)


@pytest.fixture
def raw_dir(tmp_path):
    """A raw folder holding six JPEGs in two splits, two of them identical."""
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
    photos = {
        "train/00000.jpg": make_photo(),
        "train/00001.jpg": make_photo(color=(10, 200, 10)),
        "train/00002.jpg": make_photo(box=(10, 10, 50, 50)),
        "test/00000.jpg": make_photo(width=200, height=300),
        "test/00001.jpg": make_photo(color=(200, 200, 30)),
    }
    for name, photo in photos.items():
        photo.save(tmp_path / name, quality=95)
    (tmp_path / "test/00002.jpg").write_bytes((tmp_path / "train/00000.jpg").read_bytes())
    return tmp_path


@pytest.fixture
def metadata():
    """Metadata rows matching the `raw_dir` fixture."""
    rows = [
        ("train", 0, "train/00000.jpg", 0, 100, 60, 80, 60, 240, 320),
        ("train", 1, "train/00001.jpg", 1, 100, 60, 80, 60, 240, 320),
        ("train", 2, "train/00002.jpg", 0, 10, 10, 50, 50, 240, 320),
        ("test", 0, "test/00000.jpg", 1, 100, 60, 80, 60, 300, 200),
        ("test", 1, "test/00001.jpg", 0, 100, 60, 80, 60, 240, 320),
        ("test", 2, "test/00002.jpg", 0, 100, 60, 80, 60, 240, 320),
    ]
    columns = [
        "split",
        "index",
        "filename",
        "class_index",
        "box_x",
        "box_y",
        "box_w",
        "box_h",
        "height",
        "width",
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def clusters():
    """Three well-separated Gaussian clusters in 16 dimensions, 20 points each."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16)) * 5
    labels = np.repeat(np.arange(3), 20)
    features = centers[labels] + rng.normal(size=(60, 16))
    return features.astype(np.float32), labels
