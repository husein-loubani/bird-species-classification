"""Tests for the LIME wrapper and the evidence-on-bird check."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bird_species.config import IMAGE_SIZE
from bird_species.explain import (
    box_mask,
    evidence_mask,
    evidence_on_bird,
    explain,
    model_input,
    probability_function,
)


class RedDetector(torch.nn.Module):
    """A toy classifier whose class-1 score is the amount of red in the top-left quarter."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        quarter = images[:, :, : IMAGE_SIZE // 2, : IMAGE_SIZE // 2]
        redness = (quarter[:, 0] - quarter[:, 1:].mean(dim=1)).mean(dim=(1, 2))
        return torch.stack([torch.zeros_like(redness), 20 * redness], dim=1)


def test_model_input_is_the_center_crop_as_uint8(raw_dir):
    image = model_input("test/00000.jpg", raw_dir=raw_dir)
    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and image.dtype == np.uint8


def test_probability_function_returns_distributions():
    predict = probability_function(RedDetector(), (0.0,) * 3, (1.0,) * 3, batch_size=3)
    probabilities = predict(np.zeros((5, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_lime_evidence_lands_on_the_region_the_model_uses():
    pixels = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 90, dtype=np.uint8)
    pixels[20:80, 20:80] = (230, 30, 30)
    pixels[150:210, 150:210] = (30, 30, 230)
    predict = probability_function(RedDetector(), (0.0,) * 3, (1.0,) * 3)
    explanation = explain(pixels, predict, label=1, samples=150)
    evidence = evidence_mask(explanation, 1, features=2)
    red_square = box_mask((20, 20, 80, 80))
    assert evidence_on_bird(evidence, red_square)["evidence_inside_box"] > 0.5


def test_box_mask_and_overlap_match_a_hand_count():
    bird = box_mask((0, 0, 112, 224))
    assert bird.mean() == pytest.approx(0.5)
    evidence = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    evidence[:, 56:168] = True
    result = evidence_on_bird(evidence, bird)
    assert result == {"evidence_inside_box": pytest.approx(0.5), "box_share_of_input": pytest.approx(0.5)}


def test_explain_many_caches_its_masks(tmp_path, raw_dir, monkeypatch):
    from bird_species import explain as module

    monkeypatch.setattr(module, "model_input", lambda name: model_input(name, raw_dir=raw_dir))
    predict = probability_function(RedDetector(), (0.0,) * 3, (1.0,) * 3)
    names = ["train/00000.jpg", "train/00002.jpg"]
    corners = [(0, 0, 112, 112), (0, 0, 224, 224)]
    table, masks = module.explain_many(names, [1, 1], corners, predict, tmp_path / "lime.npz", samples=60)
    again, masks_again = module.explain_many(names, [1, 1], corners, predict, tmp_path / "lime.npz", samples=60)
    assert masks.shape == (2, IMAGE_SIZE, IMAGE_SIZE) and np.array_equal(masks, masks_again)
    assert table["box_share_of_input"].round(2).tolist() == [0.25, 1.0]
    assert again["filename"].tolist() == names
