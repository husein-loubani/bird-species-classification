"""Tests for the duplicate matchers, the threshold profile and the cleaning rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bird_species.duplicates import (
    band_profile,
    duplicate_groups,
    hash_array,
    hash_pairs,
    pair_kind,
    resolve_duplicates,
    similar_pairs,
)


def test_hash_pairs_agree_with_brute_force():
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2**63, size=60, dtype=np.uint64)
    hashes[5] = hashes[7] ^ np.uint64(0b111)
    pairs = hash_pairs(hashes, radius=20, block=16)
    expected = {
        (i, j)
        for i in range(60)
        for j in range(i + 1, 60)
        if bin(int(hashes[i]) ^ int(hashes[j])).count("1") <= 20
    }
    assert set(zip(pairs["first"], pairs["second"], strict=True)) == expected
    assert pairs.loc[(pairs["first"] == 5) & (pairs["second"] == 7), "distance"].item() == 3


def test_hash_array_reads_hex_strings():
    assert hash_array(["ff", "0f"]).tolist() == [255, 15]


def test_similar_pairs_find_a_planted_copy_and_report_nearest_neighbors():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(50, 32))
    features[10] = features[3] * 2.0 + 0.01 * rng.normal(size=32)
    pairs, nearest = similar_pairs(features, floor=0.9, block=7)
    assert set(zip(pairs["first"], pairs["second"], strict=True)) == {(3, 10)}
    assert nearest[3] > 0.99 and nearest[10] > 0.99
    assert nearest[0] < 0.9


def test_pair_kind_names_the_split_combination():
    pairs = pd.DataFrame({"first": [0, 0, 1], "second": [1, 2, 2]})
    kinds = pair_kind(pairs, np.array(["train", "train", "test"]))
    assert kinds.tolist() == ["train-train", "train-test", "train-test"]


def test_band_profile_never_reads_a_withheld_label():
    pairs = pd.DataFrame(
        {"first": [0, 2, 4], "second": [1, 3, 5], "similarity": [0.99, 0.99, 0.95]}
    )
    labels = pd.Series([1, 2, 1, np.nan, 3, 3])
    profile = band_profile(pairs, "similarity", [0.9, 0.97, 1.0001], labels).set_index("band")
    top = profile.iloc[1]
    assert top["pairs"] == 2 and top["labeled_pairs"] == 1 and top["cross_label_pairs"] == 1
    assert profile.iloc[0]["cross_label_%"] == 0.0


def test_band_profile_groups_exact_values_when_given_no_bins():
    pairs = pd.DataFrame({"first": [0, 1], "second": [1, 2], "distance": [0, 2]})
    profile = band_profile(pairs, "distance", None, [0, 0, 1])
    assert profile["band"].tolist() == [0, 2]
    assert profile["cross_label_pairs"].tolist() == [0, 1]


def test_groups_follow_chains_above_the_threshold_only():
    pairs = pd.DataFrame(
        {"first": [0, 1, 3], "second": [1, 2, 4], "similarity": [0.99, 0.97, 0.90]}
    )
    groups = duplicate_groups(5, pairs, threshold=0.96)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[4]


def test_resolution_applies_one_rule_per_kind_of_duplicate():
    frame = pd.DataFrame(
        {
            "filename": [f"f{k}" for k in range(7)],
            "split": ["train", "train", "train", "test", "train", "train", "test"],
            "class_index": [0, 1, 5, 5, 2, 2, 9],
        }
    )
    # groups: {0,1} train conflict; {2,3} train copy of a test photo; {4,5} repeat; {6} alone
    data = resolve_duplicates(frame, np.array([0, 0, 1, 1, 2, 2, 3])).set_index("filename")
    assert data.loc[["f0", "f1"], "drop_reason"].eq("label conflict").all()
    assert data.loc["f2", "drop_reason"] == "copy of a test image"
    assert data.loc["f3", "drop_reason"] == ""
    assert data.loc[["f4", "f5"], "drop_reason"].tolist() == ["", "repeat within split"]
    assert data.loc["f6", "drop_reason"] == ""


def test_a_test_label_never_turns_a_copy_into_a_conflict():
    """A train photo and its test copy with different labels are a leak, not a label conflict."""
    frame = pd.DataFrame(
        {"filename": ["a", "b"], "split": ["train", "test"], "class_index": [0, 1]}
    )
    data = resolve_duplicates(frame, np.array([0, 0]))
    assert data["drop_reason"].tolist() == ["copy of a test image", ""]


def test_band_kinds_count_pairs_by_split_combination():
    from bird_species.duplicates import band_kinds

    pairs = pd.DataFrame(
        {"first": [0, 0, 2], "second": [1, 2, 3], "similarity": [0.99, 0.95, 0.99]}
    )
    kinds = band_kinds(pairs, [0.9, 0.97, 1.0001], np.array(["train", "train", "test", "test"]))
    assert kinds.to_dict("list") == {
        "test-test": [0, 1],
        "train-test": [1, 0],
        "train-train": [0, 1],
    }


def test_sample_band_stays_inside_the_band_and_sorts():
    from bird_species.duplicates import sample_band

    pairs = pd.DataFrame(
        {
            "first": range(6),
            "second": range(1, 7),
            "similarity": [0.9, 0.95, 0.96, 0.97, 0.98, 0.99],
        }
    )
    sampled = sample_band(pairs, 0.95, 0.98, count=10, seed=0)
    assert sampled["similarity"].tolist() == [0.95, 0.96, 0.97]
