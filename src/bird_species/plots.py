"""
Every figure in the notebook.

Each function returns a matplotlib Figure and never shows it, so the notebook
decides when to display and `save_figure` decides where it lands. Colors come
from the palette in `config`, every axis is labeled, and bars start at zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from skimage.segmentation import mark_boundaries

from bird_species.config import FIGURE_DIR, FIGURE_DPI, PALETTE, TARGET_ACCURACY
from bird_species.dataset import load_image

BACKBONE_COLORS = {
    "dinov2_vit_b14": PALETTE["fewshot"],
    "clip_vit_b16": PALETTE["highlight"],
    "resnet50": PALETTE["baseline"],
}


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )


def save_figure(figure: Figure, name: str, section: str, root: Path = FIGURE_DIR) -> Path:
    path = root / section / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path


def image_grid(
    panels: Sequence[dict], columns: int, title: str | None = None, panel_size: float = 2.6
) -> Figure:
    """
    Photographs in a grid, each with an optional box, evidence mask and colored frame.

    A panel is a dict with `image` and `title`, and optionally `box` (left,
    top, width, height in the image's pixels), `corners` (x0, y0, x1, y1),
    `mask` (a boolean array the size of the image) and `edge` (a frame color).
    """
    rows = int(np.ceil(len(panels) / columns))
    title_lines = max(panel.get("title", "").count("\n") + 1 for panel in panels)
    height = rows * (panel_size + 0.18 * title_lines) + 0.4
    figure, axes = plt.subplots(
        rows, columns, figsize=(panel_size * columns, height), squeeze=False
    )
    for axis in axes.ravel():
        axis.axis("off")
    for axis, panel in zip(axes.ravel(), panels, strict=False):
        image = np.asarray(panel["image"])
        if panel.get("mask") is not None:
            image = mark_boundaries(
                image, panel["mask"].astype(int), color=(1.0, 0.85, 0.0), mode="thick"
            )
        axis.imshow(image)
        if panel.get("box") is not None:
            left, top, width, height = panel["box"]
            axis.add_patch(
                Rectangle((left, top), width, height, fill=False, lw=1.5, ec=PALETTE["finetuned"])
            )
        if panel.get("corners") is not None:
            x0, y0, x1, y1 = panel["corners"]
            axis.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.5, ec=PALETTE["train"])
            )
        if panel.get("edge") is not None:
            axis.axis("on")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(panel["edge"])
                spine.set_linewidth(3)
        axis.set_title(panel.get("title", ""), fontsize=7.5)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_class_counts(train_counts: np.ndarray, test_counts: np.ndarray) -> Figure:
    """Photographs per class in each split, classes sorted by their training count."""
    order = np.argsort(-train_counts, kind="stable")
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8))
    ranks = np.arange(1, len(order) + 1)
    left.fill_between(ranks, train_counts[order], color=PALETTE["train"], alpha=0.6, label="train")
    left.plot(ranks, test_counts[order], color=PALETTE["test"], lw=0.8, label="test")
    left.set(
        title="Photographs per class, classes ranked by training count",
        xlabel="Class rank",
        ylabel="Photographs",
        ylim=(0, None),
    )
    left.legend()
    right.scatter(train_counts, test_counts, s=8, alpha=0.5, color=PALETTE["test"])
    limit = max(train_counts.max(), test_counts.max()) + 2
    right.plot(
        [0, limit], [0, limit], color=PALETTE["baseline"], lw=1, ls="--", label="equal counts"
    )
    right.set(
        title="Training against test count, one dot per class",
        xlabel="Training photographs",
        ylabel="Test photographs",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    right.legend()
    figure.tight_layout()
    return figure


def plot_variants(variants_per_species: pd.Series) -> Figure:
    """How many species are split into one, two, three or more plumage classes."""
    counts = variants_per_species.value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 3.4))
    bars = axis.bar(counts.index.astype(str), counts.to_numpy(), color=PALETTE["fewshot"])
    axis.bar_label(bars, fontsize=8)
    axis.set(
        title="Classes per species",
        xlabel="Plumage classes the species is split into",
        ylabel="Species",
        ylim=(0, counts.max() * 1.15),
    )
    figure.tight_layout()
    return figure


def plot_geometry(frame: pd.DataFrame) -> Figure:
    """Image size, shape and file size, per split."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    long_side = frame[["width", "height"]].max(axis=1)
    aspect = frame["width"] / frame["height"]
    columns = (
        (long_side, "Long side (pixels)", "Image size"),
        (aspect, "Width / height", "Image shape"),
        (frame["file_bytes"] / 1024, "File size (KB)", "JPEG size"),
    )
    for axis, (values, label, title) in zip(axes, columns, strict=True):
        for split, color in (("train", PALETTE["train"]), ("test", PALETTE["test"])):
            axis.hist(
                values[frame["split"] == split], bins=40, alpha=0.55, color=color, label=split
            )
        axis.set(title=title, xlabel=label, ylabel="Photographs")
        axis.legend()
    figure.tight_layout()
    return figure


def plot_bird_size(frame: pd.DataFrame, kept: np.ndarray) -> Figure:
    """How much of the frame the bird fills, where it sits, and how much the center crop keeps."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for split, color in (("train", PALETTE["train"]), ("test", PALETTE["test"])):
        axes[0].hist(
            frame.loc[frame["split"] == split, "box_fraction"],
            bins=40,
            alpha=0.55,
            color=color,
            label=split,
        )
    axes[0].set(
        title="Bird's share of the frame", xlabel="Box area / image area", ylabel="Photographs"
    )
    axes[0].legend()
    counts, _, _, image = axes[1].hist2d(
        frame["box_center_x"],
        frame["box_center_y"],
        bins=30,
        range=[[0, 1], [0, 1]],
        cmap="viridis",
    )
    axes[1].invert_yaxis()
    axes[1].set(
        title="Where the bird's center sits",
        xlabel="Horizontal position (0 = left edge)",
        ylabel="Vertical position (0 = top edge)",
    )
    figure.colorbar(image, ax=axes[1], label="Photographs")
    axes[2].hist(kept, bins=40, color=PALETTE["fewshot"])
    axes[2].set(
        title="Share of the bird kept by the center crop",
        xlabel="Box area inside the crop",
        ylabel="Photographs",
    )
    figure.tight_layout()
    return figure


def plot_moments(mean_image: np.ndarray, std_image: np.ndarray) -> Figure:
    """Pixel-wise mean and standard deviation of training photographs, as the models see them."""
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(mean_image, 0, 1))
    axes[0].set_title("Mean training photograph")
    shown = axes[1].imshow(std_image.mean(axis=2), cmap="magma")
    axes[1].set_title("Pixel standard deviation")
    figure.colorbar(shown, ax=axes[1], fraction=0.046, label="Std of intensity (0-1)")
    for axis in axes:
        axis.set_xlabel("Input column (pixels)")
        axis.set_ylabel("Input row (pixels)")
        axis.grid(False)
    figure.tight_layout()
    return figure


def plot_duplicate_profile(nearest: np.ndarray, profile: pd.DataFrame, threshold: float) -> Figure:
    """
    Nearest-neighbor similarity of every photograph, and how often close training pairs disagree.

    Disagreement is on the label, measured within each similarity band.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8))
    left.hist(nearest, bins=np.linspace(0.5, 1.0, 101), color=PALETTE["fewshot"], log=True)
    left.axvline(threshold, color=PALETTE["finetuned"], ls="--", label=f"threshold {threshold}")
    left.set(
        title="Each photograph's most similar other photograph",
        xlabel="Cosine similarity (DINOv2)",
        ylabel="Photographs (log scale)",
    )
    left.legend()
    labels = [f"{interval.left:.2f}" for interval in profile["band"]]
    colors = [
        PALETTE["finetuned"] if interval.left >= threshold else PALETTE["baseline"]
        for interval in profile["band"]
    ]
    bars = right.bar(labels, profile["cross_label_%"].fillna(0), color=colors)
    right.bar_label(
        bars,
        labels=[
            f"{value:.0f}%" if np.isfinite(value) else "" for value in profile["cross_label_%"]
        ],
        fontsize=7,
    )
    right.set(
        title="Training pairs whose labels differ, by similarity band",
        xlabel="Lower edge of the similarity band",
        ylabel="Pairs with different labels (%)",
        ylim=(0, max(profile["cross_label_%"].fillna(0).max() * 1.15, 1)),
    )
    right.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return figure


def plot_shot_curves(curves: pd.DataFrame, zero_shot: float) -> Figure:
    """Validation accuracy by labeled photographs per class, one line per backbone, seeds shaded."""
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    for backbone, rows in curves.groupby("backbone"):
        stats = rows.groupby("shots")["accuracy"].agg(["mean", "min", "max"])
        color = BACKBONE_COLORS.get(backbone, PALETTE["baseline"])
        axis.plot(stats.index, stats["mean"], marker="o", color=color, label=backbone)
        axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.2)
    axis.axhline(zero_shot, color=PALETTE["highlight"], ls=":", label="CLIP zero-shot (no labels)")
    axis.axhline(TARGET_ACCURACY, color=PALETTE["finetuned"], ls="--", lw=1, label="70% target")
    axis.set(
        title="Prototype classifier: accuracy by labeled photographs per class",
        xlabel="Labeled photographs per class (shots)",
        ylabel="Validation accuracy",
        ylim=(0, 1),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure


def plot_prototype_neighbors(nearest_similarity: np.ndarray, same_species: np.ndarray) -> Figure:
    """How close each class's nearest other prototype is, split by same species or not."""
    figure, axis = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(nearest_similarity.min(), nearest_similarity.max(), 30)
    axis.hist(
        [nearest_similarity[~same_species], nearest_similarity[same_species]],
        bins=bins,
        stacked=True,
        color=[PALETTE["fewshot"], PALETTE["finetuned"]],
        label=["nearest prototype is another species", "nearest prototype is the same species"],
    )
    axis.set(
        title="How close each class sits to its nearest neighbor class",
        xlabel="Cosine similarity to the nearest other prototype",
        ylabel="Classes",
    )
    axis.legend()
    figure.tight_layout()
    return figure


def plot_labeling_history(history: pd.DataFrame, random_reference: pd.DataFrame) -> Figure:
    """Accuracy by labeled photographs round by round, with random labeling as the reference."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    left.plot(
        history["labeled"],
        history["validation_accuracy"],
        marker="o",
        color=PALETTE["fewshot"],
        label="labeling loop, validation",
    )
    left.plot(
        history["labeled"],
        history["test_accuracy"],
        marker="s",
        color=PALETTE["test"],
        label="labeling loop, test",
    )
    left.plot(
        random_reference["labeled"],
        random_reference["accuracy"],
        marker="^",
        ls="--",
        color=PALETTE["baseline"],
        label="random labels, validation",
    )
    left.axhline(TARGET_ACCURACY, color=PALETTE["finetuned"], ls="--", lw=1, label="70% target")
    for _, row in history.iterrows():
        left.annotate(
            f"r{int(row['round'])}",
            (row["labeled"], row["test_accuracy"]),
            textcoords="offset points",
            xytext=(0, -12),
            fontsize=7,
            ha="center",
        )
    left.set(
        title="Accuracy as labels accumulate",
        xlabel="Labeled training photographs",
        ylabel="Accuracy",
        ylim=(0.4, 0.9),
    )
    left.legend(loc="lower right")
    rounds = history.dropna(subset=["proposed"])
    accepted = rounds["accepted"].to_numpy()
    rejected = (rounds["proposed"] - rounds["accepted"]).to_numpy()
    right.bar(rounds["round"], accepted, color=PALETTE["train"], label="confirmed, added")
    right.bar(
        rounds["round"], rejected, bottom=accepted, color=PALETTE["baseline"], label="rejected"
    )
    for _, row in rounds.iterrows():
        right.annotate(
            f"{row['precision']:.0%}",
            (row["round"], row["proposed"]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    right.set(
        title="Proposals per round (share confirmed above each bar)",
        xlabel="Round",
        ylabel="Proposed photographs",
        ylim=(0, rounds["proposed"].max() * 1.3),
    )
    right.set_xticks(rounds["round"].astype(int))
    right.legend(loc="upper right", ncols=2)
    figure.tight_layout()
    return figure


def plot_head_search(trials: pd.DataFrame) -> Figure:
    """Validation accuracy of every head-search trial against its learning rate and weight decay."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for axis, column, label in (
        (axes[0], "lr", "Learning rate"),
        (axes[1], "weight_decay", "Weight decay"),
    ):
        points = axis.scatter(
            trials[column], trials["value"], c=trials["dropout"], cmap="viridis", s=30
        )
        axis.set_xscale("log")
        axis.set(
            title=f"Head search: accuracy against {label.lower()}",
            xlabel=f"{label} (log scale)",
            ylabel="Best validation accuracy",
        )
    figure.colorbar(points, ax=axes, label="Dropout")
    return figure


def plot_learning_curves(histories: dict[str, pd.DataFrame]) -> Figure:
    """Training and validation loss and accuracy per epoch for each run."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8))
    colors = (PALETTE["fewshot"], PALETTE["finetuned"], PALETTE["highlight"], PALETTE["train"])
    for (name, history), color in zip(histories.items(), colors, strict=False):
        epochs = history["epoch"] + 1
        left.plot(epochs, history["train_loss"], color=color, ls="--", label=f"{name}, train")
        left.plot(epochs, history["val_loss"], color=color, label=f"{name}, validation")
        right.plot(epochs, history["train_accuracy"], color=color, ls="--", label=f"{name}, train")
        right.plot(epochs, history["val_accuracy"], color=color, label=f"{name}, validation")
    left.set(title="Loss per epoch", xlabel="Epoch", ylabel="Cross-entropy (with label smoothing)")
    right.set(title="Accuracy per epoch", xlabel="Epoch", ylabel="Accuracy")
    left.legend()
    right.legend()
    figure.tight_layout()
    return figure


def plot_model_comparison(summary: pd.DataFrame) -> Figure:
    """Test accuracy of every classifier with its 95% interval, colored by approach."""
    ordered = summary.sort_values("accuracy")
    figure, axis = plt.subplots(figsize=(9, 0.45 * len(ordered) + 1.2))
    colors = [PALETTE[approach] for approach in ordered["approach"]]
    errors = np.vstack(
        [
            ordered["accuracy"] - ordered["accuracy_low"],
            ordered["accuracy_high"] - ordered["accuracy"],
        ]
    )
    bars = axis.barh(ordered["model"], ordered["accuracy"], xerr=errors, color=colors, capsize=3)
    axis.bar_label(
        bars, labels=[f"{value:.1%}" for value in ordered["accuracy"]], padding=8, fontsize=8
    )
    axis.axvline(TARGET_ACCURACY, color=PALETTE["finetuned"], ls="--", lw=1)
    axis.set(
        title="Accuracy on the combined test set (bars: 95% Wilson interval)",
        xlabel="Test accuracy",
        ylabel="",
        xlim=(0, 1),
    )
    figure.tight_layout()
    return figure


def plot_accuracy_by_size(table: pd.DataFrame, colors: dict[str, str]) -> Figure:
    """Accuracy of each classifier within quartiles of the bird's share of the frame."""
    figure, axis = plt.subplots(figsize=(7.5, 4))
    labels = [
        f"{quartile}\n{share}"
        for quartile, share in zip(
            table["size_quartile"], table["bird_share_of_frame"], strict=True
        )
    ]
    for name, color in colors.items():
        axis.plot(labels, table[name], marker="o", color=color, label=name)
    axis.set(
        title="Accuracy by how much of the frame the bird fills",
        xlabel="Quartile of bird size (box area / image area)",
        ylabel="Test accuracy",
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    return figure


def plot_per_class(
    first: np.ndarray, second: np.ndarray, first_name: str, second_name: str
) -> Figure:
    """Per-class test accuracy of two classifiers, one dot per class."""
    figure, axis = plt.subplots(figsize=(5.2, 5))
    axis.scatter(first, second, s=10, alpha=0.5, color=PALETTE["fewshot"])
    axis.plot([0, 1], [0, 1], color=PALETTE["baseline"], ls="--", lw=1, label="equal accuracy")
    axis.set(
        title="Per-class accuracy",
        xlabel=f"{first_name} accuracy",
        ylabel=f"{second_name} accuracy",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.legend()
    figure.tight_layout()
    return figure


def plot_lime_overlap(table: pd.DataFrame) -> Figure:
    """
    Where LIME's evidence falls, per group of explained photographs.

    Left: the share of the evidence inside the bird's box next to the box's
    share of the input, which is what evidence scattered at random would reach.
    Right: their ratio (the lift), on a log scale, where 1 is chance.
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    groups = list(table["group"].unique())
    for position, group in enumerate(groups):
        rows = table[table["group"] == group]
        jitter = np.linspace(-0.15, 0.15, len(rows))
        left.scatter(
            position + jitter,
            rows["evidence_inside_box"],
            color=PALETTE["finetuned"],
            s=20,
            label="evidence inside the box" if position == 0 else None,
        )
        left.scatter(
            position + jitter,
            rows["box_share_of_input"],
            color=PALETTE["baseline"],
            marker="x",
            s=20,
            label="box share of the input (chance)" if position == 0 else None,
        )
        right.scatter(position + jitter, rows["lift"], color=PALETTE["highlight"], s=20)
        right.plot(
            [position - 0.25, position + 0.25],
            [rows["lift"].median()] * 2,
            color="black",
            lw=1.5,
            label="median" if position == 0 else None,
        )
    left.set_xticks(range(len(groups)), groups)
    left.set(
        title="Does the explanation sit on the bird?",
        xlabel="Explained photographs",
        ylabel="Share of input area",
        ylim=(0, 1.05),
    )
    left.legend(loc="lower left")
    right.axhline(1.0, color=PALETTE["baseline"], ls="--", lw=1, label="chance")
    right.set_yscale("log")
    right.set_xticks(range(len(groups)), groups)
    right.set(
        title="Lift: evidence inside the box over the box's share",
        xlabel="Explained photographs",
        ylabel="Lift (log scale)",
    )
    right.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_query_strategies(curves: pd.DataFrame, brief_loop: pd.DataFrame) -> Figure:
    """Validation accuracy by labels per selection strategy (seeds shaded), and the brief's loop."""
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    colors = {
        "confident": PALETTE["fewshot"],
        "uncertain": PALETTE["highlight"],
        "random": PALETTE["baseline"],
        "mixed": PALETTE["train"],
    }
    for strategy, rows in curves.groupby("strategy", sort=False):
        stats = rows.groupby("round").agg(
            labeled=("labeled", "mean"),
            mean=("accuracy", "mean"),
            low=("accuracy", "min"),
            high=("accuracy", "max"),
        )
        color = colors.get(strategy, PALETTE["test"])
        axis.plot(
            stats["labeled"],
            stats["mean"],
            marker="o",
            ms=4,
            color=color,
            label=f"{strategy}, labeler names the class",
        )
        axis.fill_between(stats["labeled"], stats["low"], stats["high"], color=color, alpha=0.2)
    axis.plot(
        brief_loop["labeled"],
        brief_loop["validation_accuracy"],
        marker="s",
        ms=4,
        ls="--",
        color=PALETTE["finetuned"],
        label="confident, labeler only confirms (the brief)",
    )
    axis.axhline(TARGET_ACCURACY, color=PALETTE["finetuned"], ls=":", lw=1, label="70% target")
    axis.set(
        title="What to show the labeler: accuracy as labels accumulate",
        xlabel="Labeled training photographs",
        ylabel="Validation accuracy",
    )
    axis.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return figure


def plot_labeler_robustness(histories: pd.DataFrame, column: str = "test_accuracy") -> Figure:
    """Accuracy (left) and share of wrong labels (right) by round, one line per labeler."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2))
    colors = (
        PALETTE["fewshot"],
        PALETTE["train"],
        PALETTE["highlight"],
        PALETTE["finetuned"],
        PALETTE["test"],
    )
    for (labeler, rows), color in zip(
        histories.groupby("labeler", sort=False), colors, strict=False
    ):
        for axis, value in ((left, column), (right, "label_noise")):
            stats = rows.groupby("round")[value].agg(["mean", "min", "max"])
            axis.plot(stats.index, stats["mean"], marker="o", ms=3, color=color, label=labeler)
            axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.2)
    left.axhline(TARGET_ACCURACY, color=PALETTE["finetuned"], ls="--", lw=1)
    left.set(
        title="Test accuracy of the brief's loop by labeler", xlabel="Round", ylabel="Test accuracy"
    )
    right.set(
        title="Share of the labeled set carrying a wrong label",
        xlabel="Round",
        ylabel="Wrong labels",
    )
    for axis in (left, right):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    left.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    return figure


def plot_reliability(tables: dict[str, pd.DataFrame]) -> Figure:
    """Accuracy against confidence per bin (left), and the share of photographs per bin (right)."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = (PALETTE["finetuned"], PALETTE["fewshot"], PALETTE["highlight"], PALETTE["baseline"])
    left.plot([0, 1], [0, 1], color=PALETTE["baseline"], ls="--", lw=1, label="perfect calibration")
    for (name, table), color in zip(tables.items(), colors, strict=False):
        left.plot(table["confidence"], table["accuracy"], marker="o", ms=4, color=color, label=name)
        right.plot(table["confidence"], table["share"], marker="o", ms=4, color=color, label=name)
    left.set(
        title="Reliability: accuracy within each confidence bin",
        xlabel="Average confidence in the bin",
        ylabel="Accuracy in the bin",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    right.set(
        title="Share of test photographs per confidence bin",
        xlabel="Average confidence in the bin",
        ylabel="Share of photographs",
        xlim=(0, 1),
    )
    left.legend(loc="lower right")
    right.legend(loc="upper left")
    figure.tight_layout()
    return figure


def plot_runs(table: pd.DataFrame) -> Figure:
    """Validation and test accuracy of each fine-tuning run, the recipe's seed range shaded."""
    figure, axis = plt.subplots(figsize=(8.5, 0.42 * len(table) + 1.4))
    positions = np.arange(len(table))
    recipe = table[table["group"] == "recipe"]
    for column, color, marker in (
        ("validation_accuracy", PALETTE["fewshot"], "o"),
        ("test_accuracy", PALETTE["finetuned"], "s"),
    ):
        axis.scatter(
            table[column],
            positions,
            color=color,
            marker=marker,
            s=28,
            label=column.replace("_", " ").replace(
                "accuracy", "accuracy (best epoch)" if "val" in column else "accuracy"
            ),
        )
        if len(recipe) > 1:
            axis.axvspan(recipe[column].min(), recipe[column].max(), color=color, alpha=0.12)
    axis.set_yticks(positions, table["run"])
    axis.invert_yaxis()
    axis.set(
        title="Fine-tuning runs (shaded: the recipe's range over three seeds)",
        xlabel="Accuracy",
        ylabel="Run",
    )
    # Below the axes, so the legend never hides a run that lands in a corner.
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2, fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def photo_panels(rows: pd.DataFrame, titles: Sequence[str], boxes: bool = True) -> list[dict]:
    """Grid panels for metadata rows: the photograph, a title, and its bird box when asked for."""
    return [
        {
            "image": load_image(row.filename),
            "title": title,
            "box": (row.box_x, row.box_y, row.box_w, row.box_h) if boxes else None,
        }
        for row, title in zip(rows.itertuples(index=False), titles, strict=True)
    ]


def verification_grid(cases: Sequence[dict]) -> Figure:
    """
    One row per proposal: the unlabeled photograph, then labeled photographs of the class it
    was given.

    A case holds `target`, `references` (filenames), `title` and `accepted`.
    The frame is green when the labeler confirmed the proposal and gray when
    it was rejected.
    """
    width = 1 + max(len(case["references"]) for case in cases)
    panels = []
    for case in cases:
        edge = PALETTE["train"] if case["accepted"] else PALETTE["baseline"]
        panels.append({"image": load_image(case["target"]), "title": case["title"], "edge": edge})
        panels.extend(
            {"image": load_image(name), "title": "labeled example"} for name in case["references"]
        )
        panels.extend(
            {"image": np.ones((8, 8, 3)), "title": ""}
            for _ in range(width - 1 - len(case["references"]))
        )
    return image_grid(panels, columns=width)


def pair_grid(pairs: pd.DataFrame, frame: pd.DataFrame, title: str) -> Figure:
    """Each sampled pair side by side, two pairs per row, with its similarity."""
    rows = frame.loc[np.ravel(pairs[["first", "second"]].to_numpy())]
    titles = [
        f"{name} ({similarity:.3f})"
        for name, similarity in zip(
            rows["filename"], np.repeat(pairs["similarity"].to_numpy(), 2), strict=True
        )
    ]
    return image_grid(photo_panels(rows, titles, boxes=False), columns=4, title=title)


def lime_grid(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    corners: Sequence[Sequence[float]],
    titles: Sequence[str],
    columns: int,
    panel_size: float = 3.6,
) -> Figure:
    """Model inputs with LIME's evidence outlined in yellow and the bird's box in green."""
    panels = [
        {"image": image, "mask": mask, "corners": box, "title": title}
        for image, mask, box, title in zip(images, masks, corners, titles, strict=True)
    ]
    return image_grid(panels, columns=columns, panel_size=panel_size)
