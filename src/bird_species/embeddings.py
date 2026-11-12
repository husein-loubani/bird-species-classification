"""
Frozen pretrained backbones as feature extractors.

Every backbone sees the same geometry (shorter side resized to 256, center
crop of 224) and its own normalization statistics, so a comparison between
them changes the network and nothing else. Features are computed once per
backbone and cached, because every few-shot experiment downstream reuses them;
the cache is keyed on the backbone, the input size and the exact list of files.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import open_clip
import timm
import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from bird_species import cache
from bird_species.config import (
    BACKBONES,
    CLIP_PRETRAINED,
    CLIP_PROMPT,
    EMBED_BATCH,
    IMAGE_SIZE,
    INTERIM_DIR,
    LOADER_WORKERS,
    RAW_DIR,
    RESIZE_SIZE,
)
from bird_species.dataset import load_image


def device() -> torch.device:
    """The machine's accelerator (an NVIDIA GPU, Apple's MPS, ...) when available, else the CPU."""
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    return accelerator if accelerator is not None else torch.device("cpu")


def eval_transform(
    mean: Sequence[float], std: Sequence[float], resize: int = RESIZE_SIZE, size: int = IMAGE_SIZE
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class ImageFiles(Dataset):
    """Photographs from the raw folder, transformed, with their labels when given."""

    def __init__(
        self,
        filenames: Sequence[str],
        transform,
        labels: Sequence[int] | None = None,
        raw_dir: Path = RAW_DIR,
    ):
        self.filenames = list(filenames)
        self.transform = transform
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.raw_dir = raw_dir

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        image = self.transform(load_image(self.filenames[index], self.raw_dir))
        if self.labels is None:
            return image
        return image, self.labels[index]


@contextmanager
def quiet_hub() -> Iterator[None]:
    """
    Hold back the Hugging Face Hub's reminder that anonymous downloads are rate-limited.

    The server sends it with every anonymous request and it says nothing about
    the weights, so it is lowered for the duration of a model load only; the
    Hub's errors still come through.
    """
    logger = logging.getLogger("huggingface_hub")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


def normalization(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The channel means and standard deviations a backbone was pretrained with."""
    if name.startswith("clip"):
        return tuple(OPENAI_DATASET_MEAN), tuple(OPENAI_DATASET_STD)
    config = timm.get_pretrained_cfg(BACKBONES[name])
    return tuple(config.mean), tuple(config.std)


class Backbone(torch.nn.Module):
    """A pretrained image encoder that maps a batch of images to one vector each."""

    def __init__(self, name: str, size: int = IMAGE_SIZE, resize: int = RESIZE_SIZE):
        super().__init__()
        self.name, self.size, self.resize = name, size, resize
        with quiet_hub():
            if name.startswith("clip"):
                model, _, _ = open_clip.create_model_and_transforms(
                    BACKBONES[name], pretrained=CLIP_PRETRAINED
                )
                self.encoder = model.visual
            else:
                extra = {"img_size": size} if "dinov2" in name else {}
                self.encoder = timm.create_model(
                    BACKBONES[name], pretrained=True, num_classes=0, **extra
                )
        self.mean, self.std = normalization(name)
        self.encoder.eval()

    @property
    def transform(self) -> transforms.Compose:
        return eval_transform(self.mean, self.std, self.resize, self.size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


@torch.no_grad()
def encode(
    backbone: Backbone,
    filenames: Sequence[str],
    batch_size: int = EMBED_BATCH,
    workers: int = LOADER_WORKERS,
    raw_dir: Path = RAW_DIR,
) -> np.ndarray:
    """One feature vector per photograph, in the order given."""
    target = device()
    backbone = backbone.to(target).eval()
    loader = DataLoader(
        ImageFiles(filenames, backbone.transform, raw_dir=raw_dir),
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    chunks = [backbone(batch.to(target)).float().cpu().numpy() for batch in loader]
    return np.concatenate(chunks).astype(np.float32)


def cached_embeddings(
    name: str,
    filenames: Sequence[str],
    cache_dir: Path = INTERIM_DIR / "embeddings",
    size: int = IMAGE_SIZE,
    resize: int = RESIZE_SIZE,
) -> np.ndarray:
    """
    The backbone's features for `filenames`, computed once and then read from disk.

    The cache is reused only for the same backbone, input size and list of
    files in the same order, so a request for other photographs recomputes
    instead of returning rows that belong to different ones.
    """
    wanted = np.asarray(filenames, dtype=str)
    path = cache_dir / (f"{name}.npz" if size == IMAGE_SIZE else f"{name}_{size}px.npz")
    key = cache.fingerprint(
        "embeddings", name, BACKBONES[name], CLIP_PRETRAINED, size, resize, wanted
    )
    if cache.is_current(path, key):
        return np.load(path, allow_pickle=False)["features"]
    features = encode(Backbone(name, size, resize), wanted)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, filenames=wanted, features=features)
    cache.record(path, key)
    return features


def prompt(class_name: str, template: str = CLIP_PROMPT) -> str:
    """
    The zero-shot caption for one class, the CLIP paper's template for bird datasets.

    The variant stays in the caption, as in `'a photo of a Common Eider (Adult
    male), a type of bird.'`, because it is the only text that can separate a
    male from a female of one species.
    """
    return template.format(name=class_name)


@torch.no_grad()
def clip_text_embeddings(class_names: Sequence[str]) -> np.ndarray:
    """CLIP's text-tower embedding of one prompt per class, for zero-shot classification."""
    with quiet_hub():
        model, _, _ = open_clip.create_model_and_transforms(
            BACKBONES["clip_vit_b16"], pretrained=CLIP_PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(BACKBONES["clip_vit_b16"])
    model.eval()
    return model.encode_text(tokenizer([prompt(name) for name in class_names])).float().numpy()
