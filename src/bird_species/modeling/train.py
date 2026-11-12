"""
The uptrained classifier: a trained head first, then the backbone fine-tuned.

Stage one trains only a linear head on frozen backbone features. The features
are cached, so a trial takes seconds and the head's hyperparameters can be
searched properly. Stage two starts from that head and unfreezes the top
transformer blocks. Starting from a trained head matters: a randomly
initialized head sends large, noisy gradients into the pretrained blocks in
the first steps and distorts the features before the head has learned
anything ("linear probing then fine-tuning", Kumar et al. 2022).

Every run keeps the weights of its best validation epoch and reloads them, so
the model that gets scored is the one that was checkpointed, not whatever the
last epoch left in memory. A finished run is reused only while its fingerprint
(training photographs, labels, starting weights, every setting and the seed)
matches the request; anything else trains it again.
"""

from __future__ import annotations

import logging
import math
import shutil
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import timm
import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from timm.optim import create_optimizer_v2
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms

from bird_species import cache
from bird_species.config import (
    BACKBONES,
    COLOR_JITTER,
    CROP_SCALE,
    FINETUNE_BATCH,
    FINETUNE_EPOCHS,
    FINETUNE_PATIENCE,
    FINETUNE_WARMUP_FRACTION,
    FINETUNE_WEIGHT_DECAY,
    HEAD_BATCH,
    HEAD_EPOCHS,
    HEAD_PATIENCE,
    HEAD_SEARCH,
    HEAD_TRIALS,
    IMAGE_SIZE,
    LABEL_SMOOTHING,
    LAYER_DECAY,
    LOADER_WORKERS,
    RANDOM_SEED,
    RESIZE_SIZE,
    UNFROZEN_BLOCKS,
)
from bird_species.embeddings import ImageFiles, eval_transform, normalization, quiet_hub

# Lightning flags conditions that do not apply to these runs: the head trains on
# the CPU on purpose (it has 426,000 weights), its data are in-memory tensors
# that need no worker processes, each run keeps its log and its checkpoint in one
# folder by design, and torch's pytree module objects to a check Lightning
# itself still makes. Each is matched by its message, so any other warning
# still surfaces.
EXPECTED_WARNINGS = (
    "GPU available but not used",
    ".*does not have many workers.*",
    "Checkpoint directory .* exists and is not empty",
    ".*isinstance\\(treespec, LeafSpec\\).*",
)


@contextmanager
def expected_warnings_held() -> Iterator[None]:
    """Hold back the expected Lightning warnings and per-run INFO lines; anything else surfaces."""
    # The device banner and the seed notice are logged at INFO for every run,
    # thirty times during the head search.
    for logger_name in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    with warnings.catch_warnings():
        for message in EXPECTED_WARNINGS:
            warnings.filterwarnings("ignore", message=message)
        yield


def train_transform(
    mean: Sequence[float],
    std: Sequence[float],
    size: int = IMAGE_SIZE,
    scale: tuple[float, float] = CROP_SCALE,
    jitter: float = COLOR_JITTER,
) -> transforms.Compose:
    """
    Random crops, mirror images and mild color shifts, all label-preserving for a bird.

    A crop keeps at least `scale[0]` of the photograph, so the bird usually
    stays in frame; a horizontal flip is safe because no field mark depends on
    which way the bird faces. Vertical flips and hue shifts are left out, since
    a bird is rarely upside down and plumage color is itself a field mark.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def balanced_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, scaled so the average photograph keeps weight one."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = len(labels) / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _feature_pairs(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed dtypes, so the same data always gives the same fingerprint."""
    return np.ascontiguousarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _file_pairs(filenames: Sequence[str], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(filenames, dtype=str), np.asarray(labels, dtype=np.int64)


def _start_fresh(run_dir: Path) -> None:
    """Remove a stale run's checkpoint and log before it is trained again."""
    shutil.rmtree(run_dir, ignore_errors=True)


class ClassifierModule(LightningModule):
    """Shared training logic: smoothed cross-entropy, with loss and accuracy logged every epoch."""

    def __init__(
        self,
        n_classes: int,
        label_smoothing: float = LABEL_SMOOTHING,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.train_accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=n_classes, average="micro")

    def step(self, batch, accuracy, stage: str):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss(logits, labels)
        accuracy(logits, labels)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, batch_size=len(labels))
        self.log(
            f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, batch_size=len(labels)
        )
        return loss

    def training_step(self, batch, batch_index):
        return self.step(batch, self.train_accuracy, "train")

    def validation_step(self, batch, batch_index):
        self.step(batch, self.val_accuracy, "val")


class LinearHead(ClassifierModule):
    """Dropout then one linear layer from backbone features to class logits."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        lr: float,
        weight_decay: float,
        dropout: float,
        max_epochs: int = HEAD_EPOCHS,
        label_smoothing: float = LABEL_SMOOTHING,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(n_classes, label_smoothing, class_weights)
        self.save_hyperparameters(ignore=["class_weights"])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(features))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return [optimizer], [schedule]


def trainer(
    max_epochs: int,
    patience: int,
    log_dir: Path,
    name: str,
    accelerator: str = "cpu",
    deterministic: bool = True,
) -> tuple[Trainer, ModelCheckpoint]:
    """A trainer that keeps the best epoch by validation accuracy and stops when it stalls."""
    checkpoint = ModelCheckpoint(
        dirpath=log_dir / name,
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    stopper = EarlyStopping(monitor="val_accuracy", mode="max", patience=patience)
    return Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint, stopper],
        logger=CSVLogger(log_dir, name=name, version=0),
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        deterministic=deterministic,
    ), checkpoint


def read_history(log_dir: Path, name: str) -> pd.DataFrame:
    """One row per epoch from a run's CSV log, train and validation metrics side by side."""
    log = pd.read_csv(log_dir / name / "version_0" / "metrics.csv")
    return log.groupby("epoch").first().reset_index()


def fit_head(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    params: dict[str, float],
    log_dir: Path,
    name: str,
    n_classes: int,
    balanced: bool = False,
    max_epochs: int = HEAD_EPOCHS,
    patience: int = HEAD_PATIENCE,
    seed: int = RANDOM_SEED,
) -> tuple[LinearHead, float]:
    """
    Train one head and return it with the weights of its best epoch, not its last.

    A finished run is reloaded from its checkpoint instead of retrained, as long
    as it was trained on the same features, labels and settings. The returned
    head carries that fingerprint as `cache_key`, so results computed from it
    can be cached against it in turn.
    """
    train, val = _feature_pairs(*train), _feature_pairs(*val)
    run_dir = log_dir / name
    weights = balanced_weights(train[1], n_classes) if balanced else None
    key = cache.fingerprint(
        "head",
        train,
        val,
        dict(params),
        n_classes,
        balanced,
        max_epochs,
        patience,
        seed,
        HEAD_BATCH,
        LABEL_SMOOTHING,
    )
    if not cache.is_current(run_dir, key):
        _start_fresh(run_dir)
        with expected_warnings_held():
            seed_everything(seed, workers=True)
            loaders = [
                DataLoader(
                    TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
                    batch_size=HEAD_BATCH,
                    shuffle=shuffle,
                )
                for (features, labels), shuffle in ((train, True), (val, False))
            ]
            module = LinearHead(
                train[0].shape[1], n_classes, max_epochs=max_epochs, class_weights=weights, **params
            )
            fit, _ = trainer(max_epochs, patience, log_dir, name)
            fit.fit(module, *loaders)
        cache.record(run_dir, key)
    best = LinearHead.load_from_checkpoint(run_dir / "best.ckpt", class_weights=weights).eval()
    best.cache_key = key
    return best, float(read_history(log_dir, name)["val_accuracy"].max())


def tune_head(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    log_dir: Path,
    n_classes: int,
    trials: int = HEAD_TRIALS,
    search: dict[str, tuple[float, float]] = HEAD_SEARCH,
    seed: int = RANDOM_SEED,
) -> optuna.Study:
    """
    Search learning rate, weight decay and dropout jointly with a seeded TPE sampler.

    The three interact: a higher learning rate tolerates stronger
    regularization, so they are searched together rather than one at a time.
    The study is stored on disk and resumed, so a rerun adds no trials; a study
    run on other features, labels or search ranges is discarded and redone.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", *search["lr"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *search["weight_decay"], log=True),
            "dropout": trial.suggest_float("dropout", *search["dropout"]),
        }
        return fit_head(train, val, params, log_dir, f"trial_{trial.number:03d}", n_classes)[1]

    log_dir.mkdir(parents=True, exist_ok=True)
    storage = log_dir / "search.db"
    train, val = _feature_pairs(*train), _feature_pairs(*val)
    key = cache.fingerprint("head search", train, val, dict(search), n_classes, seed)
    if not cache.is_current(storage, key):
        storage.unlink(missing_ok=True)
    # One INFO line per trial would repeat what the trials table shows.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="head",
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///{storage}",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    cache.record(storage, key)
    remaining = trials - len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


def create_network(
    backbone: str, n_classes: int, dropout: float, pretrained: bool = True
) -> nn.Module:
    """A timm network with a fresh `n_classes` classifier and dropout in front of it."""
    extra = {"img_size": IMAGE_SIZE} if "dinov2" in backbone else {}
    with quiet_hub():
        return timm.create_model(
            BACKBONES[backbone],
            pretrained=pretrained,
            num_classes=n_classes,
            drop_rate=dropout,
            **extra,
        )


def network_blocks(model: nn.Module) -> list[nn.Module]:
    """The network's blocks from input to output: transformer blocks or ResNet bottleneck blocks."""
    if hasattr(model, "blocks"):
        return list(model.blocks)
    return [
        block
        for stage in (model.layer1, model.layer2, model.layer3, model.layer4)
        for block in stage
    ]


def trainable_parts(model: nn.Module, unfrozen_blocks: int) -> list[nn.Module]:
    """The top `unfrozen_blocks` blocks, the final norm a transformer has, and the classifier."""
    top = network_blocks(model)[-unfrozen_blocks:] if unfrozen_blocks > 0 else []
    final_norm = [model.norm] if isinstance(getattr(model, "norm", None), nn.Module) else []
    return [*top, *final_norm, model.get_classifier()]


class FineTuner(ClassifierModule):
    """
    The pretrained backbone with the stage-one head, top blocks unfrozen.

    The earliest blocks compute generic edges and textures that transfer as
    they are; freezing them skips most of the backward pass and leaves fewer
    weights to overfit with about forty photographs per class. Learning rates
    shrink by `layer_decay` per block going down, so the blocks nearest the
    head move most and the deepest unfrozen block moves least. The same class
    fine-tunes a vision transformer (transformer blocks) or a ResNet
    (bottleneck blocks).
    """

    def __init__(
        self,
        backbone: str,
        n_classes: int,
        head_state: dict[str, torch.Tensor],
        dropout: float,
        base_lr: float,
        steps_per_epoch: int,
        unfrozen_blocks: int = UNFROZEN_BLOCKS,
        layer_decay: float = LAYER_DECAY,
        weight_decay: float = FINETUNE_WEIGHT_DECAY,
        max_epochs: int = FINETUNE_EPOCHS,
        warmup_fraction: float = FINETUNE_WARMUP_FRACTION,
        label_smoothing: float = LABEL_SMOOTHING,
        pretrained: bool = True,
    ):
        super().__init__(n_classes, label_smoothing)
        self.save_hyperparameters(ignore=["head_state", "pretrained"])
        self.model = create_network(backbone, n_classes, dropout, pretrained)
        self.model.get_classifier().load_state_dict(
            {"weight": head_state["linear.weight"], "bias": head_state["linear.bias"]}
        )
        self.model.requires_grad_(False)
        self.trainable = trainable_parts(self.model, unfrozen_blocks)
        for module in self.trainable:
            module.requires_grad_(True)

    def train(self, mode: bool = True) -> FineTuner:
        """
        Training mode everywhere except the batch norms of frozen layers.

        A batch norm in training mode keeps updating its running statistics even
        when its weights are frozen, which would quietly change the frozen
        layers' output. A transformer has no batch norm, so this only affects
        the ResNet.
        """
        super().train(mode)
        if mode:
            trainable = {
                id(module) for part in getattr(self, "trainable", []) for module in part.modules()
            }
            for module in self.model.modules():
                if (
                    isinstance(module, nn.modules.batchnorm._BatchNorm)
                    and id(module) not in trainable
                ):
                    module.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model,
            opt="adamw",
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
            layer_decay=self.hparams.layer_decay,
        )
        total = self.hparams.max_epochs * self.hparams.steps_per_epoch
        warmup = max(int(self.hparams.warmup_fraction * total), 1)

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return [optimizer], [{"scheduler": schedule, "interval": "step"}]


def trainable_share(model: FineTuner) -> float:
    """The fraction of the network's weights that fine-tuning updates."""
    parameters = list(model.model.parameters())
    return sum(p.numel() for p in parameters if p.requires_grad) / sum(
        p.numel() for p in parameters
    )


def _fit_finetuner(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    head: LinearHead,
    backbone: str,
    n_classes: int,
    base_lr: float,
    log_dir: Path,
    name: str,
    max_epochs: int,
    patience: int,
    unfrozen_blocks: int,
    workers: int,
    seed: int,
) -> None:
    seed_everything(seed, workers=True)
    mean, std = normalization(backbone)
    train_loader = DataLoader(
        ImageFiles(train[0], train_transform(mean, std), train[1]),
        batch_size=FINETUNE_BATCH,
        shuffle=True,
        num_workers=workers,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        ImageFiles(val[0], eval_transform(mean, std), val[1]),
        batch_size=FINETUNE_BATCH,
        num_workers=workers,
        persistent_workers=True,
    )
    module = FineTuner(
        backbone,
        n_classes,
        head.state_dict(),
        head.hparams.dropout,
        base_lr,
        steps_per_epoch=len(train_loader),
        max_epochs=max_epochs,
        unfrozen_blocks=unfrozen_blocks,
    )
    # Lightning picks the machine's accelerator (CUDA, Apple MPS or the CPU).
    fit, _ = trainer(max_epochs, patience, log_dir, name, accelerator="auto", deterministic=False)
    fit.fit(module, train_loader, val_loader)


def fine_tune(
    train: tuple[Sequence[str], np.ndarray],
    val: tuple[Sequence[str], np.ndarray],
    head: LinearHead,
    backbone: str,
    n_classes: int,
    base_lr: float,
    log_dir: Path,
    name: str,
    max_epochs: int = FINETUNE_EPOCHS,
    patience: int = FINETUNE_PATIENCE,
    unfrozen_blocks: int = UNFROZEN_BLOCKS,
    workers: int = LOADER_WORKERS,
    seed: int = RANDOM_SEED,
) -> tuple[FineTuner, float]:
    """
    Fine-tune from the stage-one head, or load the finished run if it is on disk.

    A full run takes about an hour on the Apple-silicon GPU, so a completed
    checkpoint is reused instead of retrained, but only when its fingerprint
    matches: the same photographs and labels, the same starting head, the same
    settings and seed. The learning curves stay in the run's `metrics.csv`, and
    the returned model carries its fingerprint as `cache_key`.
    """
    train, val = _file_pairs(*train), _file_pairs(*val)
    run_dir = log_dir / name
    start = {
        part: tensor for part, tensor in head.state_dict().items() if part.startswith("linear.")
    }
    recipe = {
        "batch": FINETUNE_BATCH,
        "warmup": FINETUNE_WARMUP_FRACTION,
        "layer_decay": LAYER_DECAY,
        "weight_decay": FINETUNE_WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "image_size": IMAGE_SIZE,
        "resize": RESIZE_SIZE,
        "crop_scale": CROP_SCALE,
        "jitter": COLOR_JITTER,
    }
    key = cache.fingerprint(
        "finetune",
        BACKBONES[backbone],
        train,
        val,
        start,
        float(head.hparams.dropout),
        n_classes,
        base_lr,
        max_epochs,
        patience,
        unfrozen_blocks,
        seed,
        recipe,
    )
    if not cache.is_current(run_dir, key):
        _start_fresh(run_dir)
        with expected_warnings_held():
            _fit_finetuner(
                train,
                val,
                head,
                backbone,
                n_classes,
                base_lr,
                log_dir,
                name,
                max_epochs,
                patience,
                unfrozen_blocks,
                workers,
                seed,
            )
        cache.record(run_dir, key)
    best = FineTuner.load_from_checkpoint(
        run_dir / "best.ckpt", head_state=head.state_dict(), pretrained=False, map_location="cpu"
    ).eval()
    best.cache_key = key
    return best, float(read_history(log_dir, name)["val_accuracy"].max())


@torch.no_grad()
def head_scores(head: LinearHead, features: np.ndarray) -> np.ndarray:
    """Softmax probabilities from a trained head on cached features."""
    return torch.softmax(head.eval()(torch.from_numpy(features)), dim=1).numpy()
