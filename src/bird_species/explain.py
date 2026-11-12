"""
LIME explanations for the fine-tuned classifier, and a check of where they point.

LIME explains one prediction at a time. It cuts the photograph into
superpixels, makes many copies with random subsets of superpixels blacked out,
asks the model for the class probability of each copy, and fits a weighted
linear model that predicts that probability from which superpixels were
present. Copies closer to the original count more. The linear model's
largest positive weights mark the regions the prediction leaned on.

The bounding boxes are never shown to the model, so they can grade the
explanation: if the evidence sits on the bird, most of its area falls inside
the box; if it sits on the perch or the water, it does not. Because a large
bird leaves little room outside its box, the share inside is read against the
box's own share of the input (the lift), not on its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torchvision import transforms

from bird_species import cache
from bird_species.config import (
    IMAGE_SIZE,
    LIME_BATCH,
    LIME_FEATURES,
    LIME_SAMPLES,
    LIME_SEGMENTS,
    RANDOM_SEED,
    RAW_DIR,
    RESIZE_SIZE,
)
from bird_species.dataset import load_image
from bird_species.embeddings import device


def model_input(
    filename: str, raw_dir: Path = RAW_DIR, resize: int = RESIZE_SIZE, crop: int = IMAGE_SIZE
) -> np.ndarray:
    """The photograph as the model sees it before normalization: resized, center-cropped, uint8."""
    geometry = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
        ]
    )
    return np.asarray(geometry(load_image(filename, raw_dir)), dtype=np.uint8)


def probability_function(
    model: torch.nn.Module,
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int = LIME_BATCH,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap the network as LIME expects: a stack of HxWx3 images in, class probabilities out."""
    target = device()
    model = model.to(target).eval()
    mean_tensor = torch.tensor(mean, device=target).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=target).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict(images: np.ndarray) -> np.ndarray:
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start : start + batch_size]))
            batch = batch.to(target).permute(0, 3, 1, 2).float() / 255.0
            outputs.append(
                torch.softmax(model((batch - mean_tensor) / std_tensor), dim=1).cpu().numpy()
            )
        return np.concatenate(outputs)

    return predict


def explain(
    image: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    label: int,
    samples: int = LIME_SAMPLES,
    seed: int = RANDOM_SEED,
):
    """LIME's explanation of the probability of `label` for one image."""
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    segmenter = SegmentationAlgorithm("quickshift", **LIME_SEGMENTS)
    # LIME 0.2 wraps its sampling loop in a progress bar and has no switch for it. In a
    # notebook the bar is a widget, not stderr, so the wrapper is swapped for a pass-through.
    with mock.patch.object(lime_image, "tqdm", lambda iterable, **_: iterable):
        explanation = explainer.explain_instance(
            image,
            predict,
            labels=(label,),
            top_labels=None,
            hide_color=0,
            num_samples=samples,
            batch_size=LIME_BATCH,
            segmentation_fn=segmenter,
            random_seed=seed,
        )
    return explanation


def evidence_mask(explanation, label: int, features: int = LIME_FEATURES) -> np.ndarray:
    """The superpixels with the largest positive weights for `label`, as a boolean mask."""
    _, mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=features, hide_rest=False
    )
    return mask.astype(bool)


def box_mask(corners: Sequence[float], size: int = IMAGE_SIZE) -> np.ndarray:
    """A boolean mask of the box given as (x0, y0, x1, y1) in input-crop pixels."""
    mask = np.zeros((size, size), dtype=bool)
    x0, y0, x1, y1 = (int(round(value)) for value in corners)
    mask[y0:y1, x0:x1] = True
    return mask


def evidence_on_bird(evidence: np.ndarray, bird: np.ndarray) -> dict[str, float]:
    """
    The share of the evidence that falls inside the bird's box, against chance.

    Evidence scattered at random would land inside the box in proportion to
    the box's area, so that area share is the baseline the overlap is read
    against. The lift divides one by the other: 1 is no better than chance,
    and its ceiling is one over the box's share, reached when all the
    evidence is inside.
    """
    inside = float((evidence & bird).sum() / max(evidence.sum(), 1))
    share = float(bird.mean())
    return {
        "evidence_inside_box": inside,
        "box_share_of_input": share,
        "lift": inside / max(share, 1e-12),
    }


def mask_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two masks: 1 when they mark the same pixels, 0 when disjoint."""
    union = (first | second).sum()
    return float((first & second).sum() / union) if union else 1.0


def explain_many(
    filenames: Sequence[str],
    labels: Sequence[int],
    corners: Sequence[Sequence[float]],
    predict: Callable[[np.ndarray], np.ndarray],
    cache_path: Path,
    model_key: str,
    samples: int = LIME_SAMPLES,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Explain each photograph's predicted class and grade the evidence against its bird box.

    Each explanation costs `samples` forward passes, so the masks and grades
    are stored. A rerun reads them back only for the same model (`model_key`,
    the fingerprint of the weights behind `predict`), photographs, classes,
    boxes, sample count and seed.
    """
    names = np.asarray(filenames, dtype=str)
    key = cache.fingerprint(
        "lime",
        model_key,
        names,
        np.asarray(labels, dtype=np.int64),
        np.asarray(corners, dtype=float),
        samples,
        seed,
        LIME_SEGMENTS,
        LIME_FEATURES,
        IMAGE_SIZE,
        RESIZE_SIZE,
    )
    if cache.is_current(cache_path, key):
        stored = np.load(cache_path, allow_pickle=False)
        table = pd.DataFrame(
            {
                column: stored[column]
                for column in ("filename", "evidence_inside_box", "box_share_of_input")
            }
        )
        return table.assign(
            lift=table["evidence_inside_box"] / table["box_share_of_input"]
        ), stored["masks"]
    rows, masks = [], []
    for filename, label, box in zip(names, labels, corners, strict=True):
        explanation = explain(model_input(filename), predict, int(label), samples, seed)
        evidence = evidence_mask(explanation, int(label))
        rows.append({"filename": filename, **evidence_on_bird(evidence, box_mask(box))})
        masks.append(evidence)
    table = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        filenames=names,
        filename=table["filename"].to_numpy(dtype=str),
        evidence_inside_box=table["evidence_inside_box"].to_numpy(),
        box_share_of_input=table["box_share_of_input"].to_numpy(),
        masks=np.stack(masks),
    )
    cache.record(cache_path, key)
    return table, np.stack(masks)
