"""Tests for the head, the fine-tuning module and their bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from bird_species.config import UNFROZEN_BLOCKS
from bird_species.modeling.train import (
    FineTuner,
    LinearHead,
    balanced_weights,
    fit_head,
    read_history,
    train_transform,
    trainable_share,
)
from tests.conftest import make_photo


def test_balanced_weights_average_one_per_photograph():
    labels = np.array([0] * 8 + [1] * 2)
    weights = balanced_weights(labels, 2)
    assert weights[1] == pytest.approx(4 * weights[0])
    assert weights[torch.from_numpy(labels)].mean().item() == pytest.approx(1.0)


def test_train_transform_returns_the_input_size():
    image = train_transform((0.5,) * 3, (0.5,) * 3)(make_photo())
    assert image.shape == (3, 224, 224)


def test_fit_head_learns_separable_features_and_reloads_the_finished_run(tmp_path, clusters):
    features, labels = clusters
    params = {"lr": 0.05, "weight_decay": 0.0, "dropout": 0.0}
    head, best = fit_head((features, labels), (features, labels), params, tmp_path, "run", 3, max_epochs=15)
    assert best == pytest.approx(1.0)
    assert isinstance(head, LinearHead) and not head.training
    stamp = (tmp_path / "run" / "best.ckpt").stat().st_mtime
    again, best_again = fit_head((features, labels), (features, labels), params, tmp_path, "run", 3)
    assert (tmp_path / "run" / "best.ckpt").stat().st_mtime == stamp, "a finished run was retrained"
    torch.testing.assert_close(again.linear.weight, head.linear.weight)
    assert best_again == best


def test_read_history_puts_train_and_validation_on_one_row(tmp_path):
    log = tmp_path / "run" / "version_0"
    log.mkdir(parents=True)
    pd.DataFrame({"epoch": [0, 0, 1, 1], "step": [5, 5, 10, 10], "val_accuracy": [0.5, None, 0.6, None],
                  "train_accuracy": [None, 0.4, None, 0.7]}).to_csv(log / "metrics.csv", index=False)
    history = read_history(tmp_path, "run")
    assert history[["val_accuracy", "train_accuracy"]].to_numpy().tolist() == [[0.5, 0.4], [0.6, 0.7]]


@pytest.fixture(scope="module")
def fine_tuner():
    head = LinearHead(768, 7, lr=1e-3, weight_decay=0.0, dropout=0.1)
    return FineTuner("dinov2_vit_b14", 7, head.state_dict(), 0.1, base_lr=1e-4, steps_per_epoch=10,
                     max_epochs=2, pretrained=False), head


def test_fine_tuner_starts_from_the_trained_head(fine_tuner):
    model, head = fine_tuner
    torch.testing.assert_close(model.model.head.weight, head.linear.weight)


def test_only_the_top_blocks_norm_and_head_are_trainable(fine_tuner):
    model, _ = fine_tuner
    blocks = model.model.blocks
    assert not any(p.requires_grad for block in blocks[:-UNFROZEN_BLOCKS] for p in block.parameters())
    assert all(p.requires_grad for block in blocks[-UNFROZEN_BLOCKS:] for p in block.parameters())
    assert not model.model.patch_embed.proj.weight.requires_grad
    assert 0.2 < trainable_share(model) < 0.5


def test_learning_rates_shrink_with_depth(fine_tuner):
    model, _ = fine_tuner
    optimizer = model.configure_optimizers()[0][0]
    scales = sorted({round(group["lr_scale"], 6) for group in optimizer.param_groups})
    assert len(scales) == UNFROZEN_BLOCKS + 1, "one rate per unfrozen block plus the head and norm"
    assert scales[-1] == pytest.approx(1.0)
    assert scales[0] == pytest.approx(0.75 ** UNFROZEN_BLOCKS)


def test_fine_tuner_outputs_class_logits(fine_tuner):
    model, _ = fine_tuner
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 7)


def test_head_scores_are_probabilities(clusters):
    from bird_species.modeling.train import head_scores

    features, _ = clusters
    head = LinearHead(16, 3, lr=1e-3, weight_decay=0.0, dropout=0.5)
    scores = head_scores(head, features)
    assert scores.shape == (60, 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(head_scores(head, features), scores, err_msg="dropout must be off when scoring")
