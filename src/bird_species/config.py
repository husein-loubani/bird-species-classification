"""
Every constant the project uses.

A reviewer asking "where does this number come from" should find the answer
here. Nothing downstream hardcodes a path, a threshold, a sample size, a
hyperparameter or a color.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nabirds"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METADATA_PATH = RAW_DIR / "metadata.csv"
CLASSES_PATH = RAW_DIR / "classes.csv"

# The brief links NABirds on Activeloop. Its "val" split is the official
# NABirds test split, and there is no third split, so the brief's "combine the
# original val and test sets" yields exactly that split.
ACTIVELOOP_SPLITS = {"train": "hub://activeloop/nabirds-dataset-train",
                     "test": "hub://activeloop/nabirds-dataset-val"}
OFFICIAL_COUNTS = {"train": 23_929, "test": 24_633}
EXPORT_WORKERS = 8

RANDOM_SEED = 42
N_JOBS = -2

# --- audit and cleaning --------------------------------------------------------

# A photograph this small cannot show the field marks that separate the
# variants, so anything below it is reported as an anomaly.
MIN_IMAGE_SIDE = 100
HASH_SIZE = 8
HASH_SEARCH_RADIUS = 8
# Pairs of DINOv2 embeddings this similar are kept for the profile; the
# duplicate threshold itself is chosen from that profile without test labels.
SIMILARITY_FLOOR = 0.80
SIMILARITY_BANDS = (0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0001)
DUPLICATE_SIMILARITY = 0.96
DUPLICATE_EXAMPLES = 4
DUPLICATE_REVIEW_BANDS = ((0.94, 0.96), (0.96, 0.97), (0.97, 1.0001))

# --- partitions ----------------------------------------------------------------

# The brief caps the starting labels at five per class. Validation is carved
# from the training pool first, never below what a five-shot start needs.
MAX_SHOTS = 5
START_SHOTS = 1
VALIDATION_FRACTION = 0.10
SHOT_SEEDS = (42, 43, 44, 45, 46)
SHOT_CURVE = (1, 2, 3, 5)

# --- exploration ---------------------------------------------------------------

SAMPLE_GRID = 12
GRID_COLUMNS = 4
BOX_EXAMPLES = 4
# Shares of the bird's box kept by the center crop that the notebook reports.
CROP_KEEP_LEVELS = (0.9, 0.5)
CLOSEST_CLASS_PAIRS = 8
# A bird filling less of the frame than this counts as small in the exploration.
SMALL_BIRD_SHARE = 0.10
MOMENT_SAMPLE = 4000
MOMENT_SIZE = 112
VARIANT_EXAMPLES = 3
VARIANTS_PER_SPECIES = 3
SIZE_BINS = 4

# --- embeddings ----------------------------------------------------------------

IMAGE_SIZE = 224
RESIZE_SIZE = 256
EMBED_BATCH = 64
LOADER_WORKERS = 6
BACKBONES = {
    "resnet50": "resnet50.tv2_in1k",
    "clip_vit_b16": "ViT-B-16-quickgelu",
    "dinov2_vit_b14": "vit_base_patch14_dinov2.lvd142m",
}
FEWSHOT_BACKBONE = "dinov2_vit_b14"
CLIP_PRETRAINED = "openai"
CLIP_PROMPT = "a photo of a {name}, a type of bird."

# --- iterative labeling --------------------------------------------------------

TARGET_ACCURACY = 0.70
PROPOSALS_PER_CLASS = 5
PROPOSAL_RANKINGS = ("margin", "similarity")
MAX_ROUNDS = 25
GALLERY_REFERENCES = 3
GALLERY_PROPOSALS = 4

# --- uptrained classifier ------------------------------------------------------

HEAD_EPOCHS = 60
HEAD_BATCH = 512
HEAD_PATIENCE = 8
HEAD_TRIALS = 30
HEAD_SEARCH = {"lr": (1e-4, 3e-2), "weight_decay": (1e-6, 1e-1), "dropout": (0.0, 0.5)}
LABEL_SMOOTHING = 0.1

FINETUNE_EPOCHS = 4
FINETUNE_BATCH = 32
FINETUNE_WARMUP_FRACTION = 0.1
FINETUNE_PATIENCE = 2
UNFROZEN_BLOCKS = 4
LAYER_DECAY = 0.75
FINETUNE_LRS = (3e-5, 1e-4)
TUNING_FRACTION = 0.25
TUNING_EPOCHS = 1
FINETUNE_WEIGHT_DECAY = 0.05
CROP_SCALE = (0.35, 1.0)
COLOR_JITTER = 0.2

# --- evaluation and explanation ------------------------------------------------

CONFIDENCE_LEVEL = 0.95
TOP_K = 5
TIMING_SAMPLE = 100
CONFUSED_PAIRS = 10
# Bands of similarity to the nearest training photograph, for the validation/test comparison.
CLOSENESS_BINS = (0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001)
LIME_EXAMPLES = 6
LIME_COLUMNS = 3
LIME_METRIC_EXAMPLES = 20
LIME_SAMPLES = 1000
LIME_FEATURES = 5
LIME_BATCH = 50
LIME_SEGMENTS = {"kernel_size": 4, "max_dist": 200, "ratio": 0.2}

PALETTE = {
    "fewshot": "#4C72B0",
    "finetuned": "#C44E52",
    "baseline": "#8C8C8C",
    "highlight": "#DD8452",
    "train": "#55A868",
    "test": "#8172B3",
}
FIGURE_DPI = 120
