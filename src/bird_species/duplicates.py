"""
Finding the same photograph, or the same moment, saved more than once.

Identical files share a SHA-256. The harder cases are one photograph saved at
another size, crop or JPEG quality, and burst frames: two exposures of the
same bird in the same scene a fraction of a second apart. Neither shares a
single byte with its twin, and either one leaks if its twins sit on both sides
of the train/test boundary.

Two matchers are measured. The perceptual hash (pHash) marks which of the 64
lowest-frequency DCT coefficients of a 32x32 grayscale thumbnail lie above
their median, so it describes the layout of light and dark. The embedding matcher compares frozen
DINOv2 features by cosine similarity, so it describes content. Both reduce to
a list of close pairs, and the same label-free and train-only checks decide
how close counts as a duplicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bird_species.config import DUPLICATE_SIMILARITY, HASH_SEARCH_RADIUS, SIMILARITY_FLOOR

BLOCK_ROWS = 2048


def hash_array(hex_hashes) -> np.ndarray:
    """Hex pHash strings as unsigned 64-bit integers."""
    return np.array([int(value, 16) for value in hex_hashes], dtype=np.uint64)


def hash_pairs(
    hashes: np.ndarray, radius: int = HASH_SEARCH_RADIUS, block: int = BLOCK_ROWS // 4
) -> pd.DataFrame:
    """
    Every pair (first < second) whose hashes differ in at most `radius` bits.

    About 1.2 billion comparisons, done in row blocks with a vectorized XOR and
    popcount so memory stays bounded.
    """
    firsts, seconds, distances = [], [], []
    for start in range(0, len(hashes), block):
        distance = np.bitwise_count(hashes[start : start + block, None] ^ hashes[None, :])
        local, other = np.nonzero(distance <= radius)
        first = local + start
        keep = other > first
        firsts.append(first[keep])
        seconds.append(other[keep])
        distances.append(distance[local[keep], other[keep]])
    return pd.DataFrame(
        {
            "first": np.concatenate(firsts),
            "second": np.concatenate(seconds),
            "distance": np.concatenate(distances).astype(int),
        }
    )


def unit_rows(features: np.ndarray) -> np.ndarray:
    return (features / np.linalg.norm(features, axis=1, keepdims=True)).astype(np.float32)


def similar_pairs(
    features: np.ndarray, floor: float = SIMILARITY_FLOOR, block: int = BLOCK_ROWS
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Every pair with cosine similarity of at least `floor`, and each image's nearest-neighbor
    similarity.

    The full similarity matrix of ~48,000 images would need 9 GB, so it is
    built one row block at a time and only the close pairs are kept.
    """
    unit = unit_rows(features)
    firsts, seconds, similarities = [], [], []
    nearest = np.empty(len(unit), dtype=np.float32)
    for start in range(0, len(unit), block):
        similarity = unit[start : start + block] @ unit.T
        rows = np.arange(len(similarity))
        similarity[rows, rows + start] = -1.0
        nearest[start : start + len(similarity)] = similarity.max(axis=1)
        local, other = np.nonzero(similarity >= floor)
        first = local + start
        keep = other > first
        firsts.append(first[keep])
        seconds.append(other[keep])
        similarities.append(similarity[local[keep], other[keep]])
    pairs = pd.DataFrame(
        {
            "first": np.concatenate(firsts),
            "second": np.concatenate(seconds),
            "similarity": np.concatenate(similarities),
        }
    )
    return pairs, nearest


def pair_kind(pairs: pd.DataFrame, split: np.ndarray) -> np.ndarray:
    """Whether each pair sits inside the training split, inside the test split, or across them."""
    first, second = split[pairs["first"]], split[pairs["second"]]
    return np.where(
        first == second, np.char.add(first.astype(str), "-" + second.astype(str)), "train-test"
    )


def band_profile(pairs: pd.DataFrame, column: str, bins, labels) -> pd.DataFrame:
    """
    For each band of closeness, how many pairs there are and how many join different labels.

    `bins` are half-open band edges, or None to group on the exact values of
    `column` (a Hamming distance takes few enough values for that).

    `labels` holds missing values for withheld images, and a pair is checked
    only when both of its labels are present, so passing the training labels
    alone keeps every test label out of the threshold choice. Two copies of one
    photograph cannot show two different categories, so the cross-label share
    is a floor on the matcher's error rate in that band.
    """
    values = pd.Series(labels, dtype="float64").to_numpy()
    first, second = values[pairs["first"]], values[pairs["second"]]
    known = ~np.isnan(first) & ~np.isnan(second)
    band = pairs[column] if bins is None else pd.cut(pairs[column], bins, right=False)
    frame = pd.DataFrame({"band": band, "known": known, "cross": known & (first != second)})
    profile = frame.groupby("band", observed=False).agg(
        pairs=("known", "size"), labeled_pairs=("known", "sum"), cross_label_pairs=("cross", "sum")
    )
    profile["cross_label_%"] = (
        100
        * profile["cross_label_pairs"]
        / profile["labeled_pairs"].where(profile["labeled_pairs"] > 0)
    ).round(1)
    return profile.reset_index()


def duplicate_groups(
    count: int, pairs: pd.DataFrame, threshold: float = DUPLICATE_SIMILARITY
) -> np.ndarray:
    """
    A group id per image: connected components of the "same photograph" graph.

    If A matches B and B matches C, all three share a group, because keeping C
    apart from A would still leave the same picture on both sides through B.
    """
    linked = pairs[pairs["similarity"] >= threshold]
    graph = coo_matrix(
        (np.ones(len(linked)), (linked["first"], linked["second"])), shape=(count, count)
    )
    return connected_components(graph, directed=False)[1]


def resolve_duplicates(frame: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    """
    Decide which photographs leave the data, one rule per kind of duplicate.

    - A training group whose members carry different labels is dropped: the
      same picture cannot be two categories, so at least one label is wrong and
      nothing says which. Only training labels are read for this, so no test
      label shapes the cleaning.
    - A group that reaches into the test set loses its training members, which
      removes the leak and keeps every distinct test photograph.
    - Inside one split, the first member of each group stays and the copies go,
      so no photograph counts twice.
    """
    data = frame.assign(group=groups).reset_index(drop=True)
    data["drop_reason"] = ""
    sizes = data.groupby("group")["filename"].transform("size")
    repeated = data[sizes > 1]
    train_repeats = repeated[repeated["split"] == "train"]
    labels_per_group = train_repeats.groupby("group")["class_index"].nunique()
    conflict_groups = set(labels_per_group.index[labels_per_group > 1])
    test_groups = set(repeated.loc[repeated["split"] == "test", "group"])

    is_train = data["split"] == "train"
    data.loc[is_train & data["group"].isin(conflict_groups), "drop_reason"] = "label conflict"
    leak = is_train & data["group"].isin(test_groups) & (data["drop_reason"] == "")
    data.loc[leak, "drop_reason"] = "copy of a test image"
    remaining = data[data["drop_reason"] == ""]
    extra = remaining.duplicated(["split", "group"], keep="first")
    data.loc[extra[extra].index, "drop_reason"] = "repeat within split"
    return data


def sample_band(
    pairs: pd.DataFrame, low: float, high: float, count: int, seed: int
) -> pd.DataFrame:
    """Up to `count` random pairs with similarity in [low, high), for looking at them."""
    band = pairs[(pairs["similarity"] >= low) & (pairs["similarity"] < high)]
    return band.sample(min(count, len(band)), random_state=seed).sort_values("similarity")


def band_kinds(pairs: pd.DataFrame, bins, split: np.ndarray) -> pd.DataFrame:
    """Pair counts per similarity band, split into train-train, test-test and train-test pairs."""
    band = pd.cut(pairs["similarity"], bins, right=False)
    return pd.crosstab(band, pair_kind(pairs, split), dropna=False).reset_index(drop=True)
