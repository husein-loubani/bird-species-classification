"""Smoke tests: every figure renders from small inputs, returns a Figure, and labels its axes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from bird_species import plots
from tests.conftest import make_photo


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def labeled(figure: Figure) -> bool:
    """Every axis that holds data has an x-label and a y-label."""
    axes = [axis for axis in figure.axes if axis.has_data() and axis.get_label() != "<colorbar>"]
    return all(axis.get_xlabel() and axis.get_ylabel() for axis in axes)


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({"split": ["train", "test"] * (n // 2), "width": rng.integers(300, 1024, n),
                         "height": rng.integers(300, 1024, n), "file_bytes": rng.integers(10_000, 300_000, n),
                         "box_fraction": rng.uniform(0.01, 0.9, n), "box_center_x": rng.uniform(0, 1, n),
                         "box_center_y": rng.uniform(0, 1, n)})


def test_eda_figures(frame):
    figures = [plots.plot_class_counts(np.array([5, 9, 4]), np.array([6, 8, 9])),
               plots.plot_variants(pd.Series([1, 1, 2, 3])),
               plots.plot_geometry(frame),
               plots.plot_bird_size(frame, np.linspace(0.5, 1, len(frame))),
               plots.plot_moments(np.full((8, 8, 3), 0.5), np.full((8, 8, 3), 0.1))]
    assert all(isinstance(figure, Figure) and labeled(figure) for figure in figures)


def test_duplicate_profile_marks_bands_above_the_threshold():
    bands = pd.IntervalIndex.from_breaks([0.9, 0.95, 0.97, 1.0], closed="left")
    profile = pd.DataFrame({"band": bands, "cross_label_%": [12.0, 1.0, np.nan]})
    figure = plots.plot_duplicate_profile(np.random.default_rng(0).uniform(0.6, 1, 100), profile, 0.96)
    assert labeled(figure)


def test_image_grid_draws_boxes_masks_and_frames():
    mask = np.zeros((240, 320), dtype=bool)
    mask[50:100, 50:100] = True
    panels = [{"image": make_photo(), "title": "a", "box": (100, 60, 80, 60)},
              {"image": np.asarray(make_photo()), "title": "b", "mask": mask, "corners": (0, 0, 50, 50),
               "edge": "green"}]
    figure = plots.image_grid(panels, columns=3, title="grid")
    assert len(figure.axes) == 3 and figure.axes[0].patches


def test_fewshot_figures():
    curves = pd.DataFrame({"backbone": ["dinov2_vit_b14"] * 4 + ["resnet50"] * 4, "shots": [1, 1, 5, 5] * 2,
                           "seed": [0, 1] * 4, "accuracy": [0.5, 0.55, 0.7, 0.72, 0.2, 0.22, 0.3, 0.33]})
    history = pd.DataFrame({"round": [0, 1, 2], "labeled": [100, 300, 500], "validation_accuracy": [0.5, 0.6, 0.7],
                            "test_accuracy": [0.5, 0.6, 0.71], "proposed": [np.nan, 250, 240],
                            "accepted": [np.nan, 200, 190], "precision": [np.nan, 0.8, 0.79]})
    reference = pd.DataFrame({"labeled": [100, 500], "accuracy": [0.5, 0.75]})
    figures = [plots.plot_shot_curves(curves, zero_shot=0.4),
               plots.plot_prototype_neighbors(np.linspace(0.3, 0.9, 20), np.arange(20) % 2 == 0),
               plots.plot_labeling_history(history, reference)]
    assert all(labeled(figure) for figure in figures)


def test_training_figures():
    trials = pd.DataFrame({"lr": [1e-3, 1e-2], "weight_decay": [1e-4, 1e-2], "dropout": [0.1, 0.3],
                           "value": [0.8, 0.82]})
    history = pd.DataFrame({"epoch": [0, 1], "train_loss": [2.0, 1.5], "val_loss": [2.1, 1.7],
                            "train_accuracy": [0.5, 0.6], "val_accuracy": [0.55, 0.62]})
    assert labeled(plots.plot_head_search(trials))
    assert labeled(plots.plot_learning_curves({"run": history}))


def test_evaluation_figures():
    summary = pd.DataFrame({"model": ["a", "b"], "approach": ["fewshot", "finetuned"], "accuracy": [0.7, 0.85],
                            "accuracy_low": [0.69, 0.84], "accuracy_high": [0.71, 0.86]})
    sizes = pd.DataFrame({"size_quartile": ["Q1", "Q2"], "bird_share_of_frame": ["1% to 10%", "10% to 90%"],
                          "a": [0.6, 0.8], "b": [0.8, 0.9]})
    lime = pd.DataFrame({"group": ["best", "best", "worst"], "evidence_inside_box": [0.9, 0.8, 0.3],
                         "box_share_of_input": [0.3, 0.4, 0.2]})
    figures = [plots.plot_model_comparison(summary),
               plots.plot_accuracy_by_size(sizes, {"a": "red", "b": "blue"}),
               plots.plot_per_class(np.array([0.5, 1.0]), np.array([0.8, 1.0]), "a", "b"),
               plots.plot_lime_overlap(lime)]
    assert all(isinstance(figure, Figure) for figure in figures)
    assert all(labeled(figure) for figure in figures[1:])


def test_save_figure_writes_a_png(tmp_path):
    path = plots.save_figure(plots.plot_variants(pd.Series([1, 2])), "variants", "eda", root=tmp_path)
    assert path.exists() and path.suffix == ".png"


def test_photo_grids_load_images_and_draw_boxes(raw_dir, monkeypatch):
    monkeypatch.setattr(plots, "load_image", lambda name: make_photo())
    rows = pd.DataFrame({"filename": ["a", "b", "c", "d"], "box_x": [1] * 4, "box_y": [1] * 4, "box_w": [5] * 4,
                         "box_h": [5] * 4})
    pairs = pd.DataFrame({"first": [0, 2], "second": [1, 3], "similarity": [0.97, 0.99]})
    assert len(plots.pair_grid(pairs, rows, "pairs").axes) == 4
    cases = [{"target": "a", "references": ["b", "c"], "title": "t", "accepted": True},
             {"target": "d", "references": ["b"], "title": "t", "accepted": False}]
    assert len(plots.verification_grid(cases).axes) == 6
    mask = np.zeros((240, 320), dtype=bool)
    figure = plots.lime_grid([np.asarray(make_photo())], [mask], [(0, 0, 10, 10)], ["x"], columns=1)
    assert figure.axes[0].patches
