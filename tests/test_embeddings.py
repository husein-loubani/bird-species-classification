"""Tests for the image pipeline, the feature cache and the zero-shot captions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bird_species import embeddings
from bird_species.config import BACKBONES, IMAGE_SIZE
from bird_species.embeddings import (
    ImageFiles,
    cached_embeddings,
    eval_transform,
    normalization,
    prompt,
)


def test_eval_transform_produces_the_model_input_size(raw_dir):
    dataset = ImageFiles(["train/00000.jpg", "test/00000.jpg"], eval_transform((0.5,) * 3, (0.5,) * 3),
                         labels=[3, 4], raw_dir=raw_dir)
    image, label = dataset[1]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and label == 4
    assert len(dataset) == 2


def test_image_files_without_labels_return_only_the_tensor(raw_dir):
    image = ImageFiles(["train/00001.jpg"], eval_transform((0.5,) * 3, (0.5,) * 3), raw_dir=raw_dir)[0]
    assert isinstance(image, torch.Tensor)


@pytest.mark.parametrize("name", list(BACKBONES))
def test_every_backbone_has_three_channel_statistics(name):
    mean, std = normalization(name)
    assert len(mean) == 3 and len(std) == 3 and all(0 < value < 1 for value in mean + std)


def test_feature_cache_is_reused_only_for_the_same_files(tmp_path, monkeypatch):
    calls = []

    def fake_encode(backbone, filenames):
        calls.append(list(filenames))
        return np.arange(len(filenames), dtype=np.float32)[:, None] * np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(embeddings, "Backbone", lambda name: name)
    monkeypatch.setattr(embeddings, "encode", fake_encode)
    first = cached_embeddings("resnet50", ["a.jpg", "b.jpg"], cache_dir=tmp_path)
    again = cached_embeddings("resnet50", ["a.jpg", "b.jpg"], cache_dir=tmp_path)
    other = cached_embeddings("resnet50", ["b.jpg", "a.jpg"], cache_dir=tmp_path)
    assert len(calls) == 2, "a matching cache must not recompute, a reordered request must"
    np.testing.assert_array_equal(first, again)
    assert other.shape == (2, 4)


def test_prompts_keep_the_variant():
    assert prompt("Common Eider (Adult male)") == "a photo of a Common Eider (Adult male), a type of bird."
