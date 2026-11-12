"""Tests for the episodic training of a projection on top of frozen features."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bird_species.episodic import (
    ResidualProjection,
    episode_loss,
    project,
    prototype_accuracy,
    sample_episode,
    split_species,
    train_projection,
)


def test_species_split_keeps_every_plumage_of_a_species_together():
    species = np.array(["a", "a", "b", "c", "c", "d", "e", "f", "g", "h", "i", "j"])
    roles = split_species(species, (0.6, 0.2, 0.2), seed=0)
    for name in np.unique(species):
        assert len(set(roles[species == name])) == 1
    counts = {
        role: len(np.unique(species[roles == role])) for role in ("base", "validation", "novel")
    }
    assert counts == {"base": 6, "validation": 2, "novel": 2}


def test_projection_starts_as_the_identity():
    features = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    np.testing.assert_allclose(project(ResidualProjection(8, 4), features), features, atol=1e-6)
    assert project(None, features) is features


def test_an_episode_draws_distinct_photographs_per_class():
    rows_by_class = [np.arange(10 * c, 10 * c + 10) for c in range(6)]
    support, queries = sample_episode(
        rows_by_class, ways=4, shots=2, queries=3, rng=np.random.default_rng(0)
    )
    assert support.shape == (4, 2) and queries.shape == (4, 3)
    for picked in np.hstack([support, queries]):
        assert len(set(picked)) == 5 and len({row // 10 for row in picked}) == 1


def test_episode_loss_is_small_when_classes_are_separated(clusters):
    features, labels = clusters
    data = torch.from_numpy(features)
    support = np.stack([np.flatnonzero(labels == c)[:3] for c in range(3)])
    queries = np.stack([np.flatnonzero(labels == c)[3:6] for c in range(3)])
    loss = episode_loss(ResidualProjection(16, 8), data, support, queries)
    assert loss.item() < np.log(3), "better than guessing among three classes"


def test_training_never_returns_a_projection_worse_than_its_start(clusters):
    features, labels = clusters
    rng = np.random.default_rng(1)
    noisy = features + rng.normal(scale=4.0, size=features.shape).astype(np.float32)
    support = np.concatenate([np.flatnonzero(labels == c)[:2] for c in range(3)])
    asked = np.setdiff1d(np.arange(len(labels)), support)
    start = prototype_accuracy(noisy, labels, support, asked)
    projection, history = train_projection(
        noisy,
        labels,
        lambda p: prototype_accuracy(noisy, labels, support, asked, p),
        ways=3,
        shots=2,
        queries=3,
        episodes=40,
        check_every=10,
        hidden=8,
        seed=0,
    )
    assert history["episode"].tolist() == [0, 10, 20, 30, 40]
    assert prototype_accuracy(noisy, labels, support, asked, projection) >= start
    assert history["monitor"].iloc[0] == pytest.approx(start)


def test_prototype_accuracy_competes_only_among_the_support_classes(clusters):
    features, labels = clusters
    support = np.array([0, 1, 20, 21])
    asked = np.arange(2, 20)
    assert prototype_accuracy(features, labels, support, asked) == 1.0
