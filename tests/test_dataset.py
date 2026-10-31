"""Tests for loading, auditing, box geometry and the partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bird_species.dataset import (
    assign_partitions,
    audit_file,
    audit_images,
    audit_summary,
    box_in_crop,
    box_kept_by_crop,
    carve_validation,
    class_table,
    clip_boxes,
    draw_shots,
    load_image,
    partition_summary,
    pixel_moments,
    split_class_name,
)
from tests.conftest import make_photo


@pytest.mark.parametrize(("name", "expected"), [
    ("Common Eider (Adult male)", ("Common Eider", "Adult male")),
    ("Yellow-rumped Warbler (Winter/juvenile Audubon's)", ("Yellow-rumped Warbler", "Winter/juvenile Audubon's")),
    ("Wild Turkey", ("Wild Turkey", "")),
])
def test_class_names_split_into_species_and_variant(name, expected):
    assert split_class_name(name) == expected


def test_class_table_reindexes_only_the_labels_in_use():
    names = ["Birds", "Ducks", "Mallard (Male)", "Mallard (Female)", "Loon"]
    table = class_table(names, [3, 2, 4, 3])
    assert table["label"].tolist() == [2, 3, 4]
    assert table["class_index"].tolist() == [0, 1, 2]
    assert table["species"].tolist() == ["Mallard", "Mallard", "Loon"]


def test_audit_decodes_a_good_file_and_fingerprints_it(raw_dir):
    row = audit_file("train/00000.jpg", raw_dir)
    assert row["decodes"] and row["mode"] == "RGB"
    assert (row["decoded_width"], row["decoded_height"]) == (320, 240)
    assert len(row["phash"]) == 16 and len(row["sha256"]) == 64


def test_audit_flags_a_truncated_jpeg(raw_dir):
    data = (raw_dir / "train/00001.jpg").read_bytes()
    (raw_dir / "train/broken.jpg").write_bytes(data[: len(data) // 3])
    assert not audit_file("train/broken.jpg", raw_dir)["decodes"]


def test_audit_summary_counts_identical_files_and_bad_boxes(raw_dir, metadata):
    metadata.loc[4, "box_w"] = 400
    summary = audit_summary(audit_images(metadata, raw_dir, n_jobs=1)).set_index("split")
    assert summary.loc["test", "identical_bytes"] == 1
    assert summary.loc["test", "box_outside"] == 1
    assert summary.loc["train", "fails_decode"] == 0


def test_load_image_returns_rgb_for_a_grayscale_file(tmp_path):
    make_photo().convert("L").save(tmp_path / "gray.jpg")
    assert load_image("gray.jpg", tmp_path).mode == "RGB"


def test_clip_boxes_cuts_an_overshooting_box_at_the_edge(metadata):
    metadata.loc[0, "box_w"] = 1000
    clipped = clip_boxes(metadata)
    assert clipped.loc[0, "box_x"] + clipped.loc[0, "box_w"] == metadata.loc[0, "width"]
    assert clipped.loc[1, "box_fraction"] == pytest.approx(80 * 60 / (320 * 240))
    assert clipped.loc[1, "box_center_x"] == pytest.approx(140 / 320)


def test_box_in_crop_matches_a_hand_computed_resize_and_crop():
    # A 512x256 photograph is already 256 on its short side, so the scale is 1
    # and the 224 crop starts 144 pixels in and 16 pixels down.
    assert box_in_crop((144, 16, 224, 224), 512, 256) == (0.0, 0.0, 224.0, 224.0)
    assert box_in_crop((0, 0, 200, 100), 512, 256) == (0.0, 0.0, 56.0, 84.0)


def test_box_kept_by_crop_is_one_inside_and_a_half_when_half_is_cut():
    frame = pd.DataFrame({"width": [512, 512], "height": [256, 256], "box_x": [200, 32],
                          "box_y": [50, 50], "box_w": [100, 224], "box_h": [100, 100]})
    np.testing.assert_allclose(box_kept_by_crop(frame), [1.0, 0.5])


def test_validation_keeps_duplicate_groups_whole_and_leaves_enough_shots():
    pool = pd.DataFrame({"class_index": [0] * 20 + [1] * 4, "group": list(range(10)) * 2 + [20, 21, 22, 23]})
    marked = carve_validation(pool, fraction=0.3, shots=5, seed=0)
    by_group = pool.assign(validation=marked).groupby("group")["validation"].nunique()
    assert (by_group == 1).all(), "a duplicate group straddles the boundary"
    assert not marked[pool["class_index"] == 1].any(), "a four-photo class cannot spare any"
    assert marked[pool["class_index"] == 0].sum() == 6


def test_draw_shots_takes_at_most_k_per_class_one_per_group():
    pool = pd.DataFrame({"class_index": [0] * 6 + [1] * 2, "group": [0, 0, 1, 2, 3, 4, 5, 6]})
    chosen = draw_shots(pool, shots=5, seed=1)
    counts = pool.loc[chosen].groupby("class_index").size()
    assert counts.to_dict() == {0: 5, 1: 2}
    assert pool.loc[chosen, "group"].is_unique
    assert list(chosen) == list(draw_shots(pool, shots=5, seed=1))


def test_partitions_seal_the_test_split_and_carve_validation_from_train():
    frame = pd.DataFrame({"split": ["train"] * 40 + ["test"] * 5, "class_index": [0] * 45,
                          "group": range(45)})
    data = assign_partitions(frame, seed=0)
    assert (data.loc[data["split"] == "test", "partition"] == "test").all()
    summary = partition_summary(data).set_index("partition")
    assert summary.loc["validation", "images"] == 4
    assert summary.loc["train", "images"] == 36


def test_pixel_moments_of_identical_photographs_have_zero_spread(tmp_path):
    for k in range(3):
        make_photo(box=(0, 0, 1, 1), box_color=(30, 120, 200)).save(tmp_path / f"{k}.png")
    mean, std = pixel_moments([f"{k}.png" for k in range(3)], size=16, raw_dir=tmp_path, n_jobs=1)
    np.testing.assert_allclose(mean[8, 8], np.array([30, 120, 200]) / 255, atol=1e-6)
    assert std.max() < 1e-6


def test_variant_examples_take_species_split_into_enough_variants():
    from bird_species.dataset import variant_examples

    classes = pd.DataFrame({"class_index": range(6),
                            "class_name": ["A (Male)", "A (Female)", "B", "C (Male)", "C (Female)", "C (Juvenile)"],
                            "species": ["A", "A", "B", "C", "C", "C"]})
    frame = pd.DataFrame({"class_index": [0, 1, 2, 3, 4, 5, 5], "filename": list("abcdefg")})
    examples = variant_examples(frame, classes, species_count=2, per_species=2, seed=0)
    assert examples["class_name"].tolist() == ["A (Male)", "A (Female)", "C (Male)", "C (Female)"]


def test_write_processed_writes_one_csv_per_table(tmp_path):
    from bird_species.dataset import write_processed

    written = write_processed({"b": pd.DataFrame({"x": [1]}), "a": pd.DataFrame({"y": [2]})}, tmp_path)
    assert written == ["a.csv", "b.csv"] and (tmp_path / "a.csv").exists()
