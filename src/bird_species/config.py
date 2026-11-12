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
ACTIVELOOP_SPLITS = {
    "train": "hub://activeloop/nabirds-dataset-train",
    "test": "hub://activeloop/nabirds-dataset-val",
}
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
# Simulated labelers: the chance of confirming a wrong proposal (and of missing a right one),
# and the chance of confirming a wrong proposal whose class looks like the true one.
LABELER_SLIPS = (0.05, 0.10, 0.20)
LOOKALIKE_SLIP = 0.5
LABELER_SEEDS = (42, 43, 44, 45, 46)
# The brief's loop with each simulated labeler runs this many rounds, past the 70% target.
LABELER_ROUNDS = 12
# Selection strategies compared with a labeler who names every photograph's class.
QUERY_STRATEGIES = ("confident", "uncertain", "random", "mixed")
QUERY_PER_CLASS = 2
QUERY_ROUNDS = 8
QUERY_SEEDS = (42, 43, 44)
# Rounds at which the strategies are compared in the notebook's table.
QUERY_REPORT_ROUNDS = (2, 4, 8)

# --- training the embedding on episodes ---------------------------------------

SPECIES_SPLIT = (0.6, 0.2, 0.2)
SPLIT_SEEDS = (42, 43, 44)
EPISODE_WAYS = 50
EPISODE_SHOTS = 5
EPISODE_QUERIES = 5
# Episodes drawn from five labels per class split them into two support and three query photographs.
FEW_LABEL_SHOTS = 2
FEW_LABEL_QUERIES = 3
EPISODES = 1000
EPISODE_CHECK = 25
PROJECTION_HIDDEN = 512
PROJECTION_LR = 3e-4
PROJECTION_WEIGHT_DECAY = 1e-4

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
FINETUNE_LR = 3e-5
# Rates for the short check: one epoch on a quarter of the training set.
FINETUNE_LRS = (1e-5, 3e-5, 1e-4, 3e-4)
TUNING_FRACTION = 0.25
TUNING_EPOCHS = 1
FINETUNE_WEIGHT_DECAY = 0.05
FINETUNE_SEEDS = (42, 43, 44)
# Each variant changes one setting of the recipe and keeps seed 42.
FINETUNE_VARIANTS = {
    "epochs_8": {"max_epochs": 8},
    "blocks_12": {"unfrozen_blocks": 12},
    "lr_1e-05": {"base_lr": 1e-5},
    "lr_0.0001": {"base_lr": 1e-4},
}
# ResNet-50's last stage (layer4) holds its last three bottleneck blocks.
RESNET_UNFROZEN_BLOCKS = 3
RESNET_LRS = (1e-4, 3e-4, 1e-3)
# A higher input resolution, tried on frozen features before paying for it in fine-tuning.
PROBE_SIZE = 448
PROBE_RESIZE = 512
CROP_SCALE = (0.35, 1.0)
COLOR_JITTER = 0.2

# --- evaluation and explanation ------------------------------------------------

CONFIDENCE_LEVEL = 0.95
TOP_K = 5
# Equal-width confidence bins for the calibration error, as in Guo et al. (2017).
CALIBRATION_BINS = 15
TEMPERATURE_BOUNDS = (0.05, 20.0)
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
# Random test photographs explained from the smallest and the largest quartile of bird size.
LIME_SIZE_EXAMPLES = 20
# Extra sampling seeds for re-explaining the photographs shown, to check the evidence is stable.
LIME_SEEDS = (43, 44)

PALETTE = {
    "fewshot": "#4C72B0",
    "finetuned": "#C44E52",
    "baseline": "#8C8C8C",
    "highlight": "#DD8452",
    "train": "#55A868",
    "test": "#8172B3",
}
FIGURE_DPI = 120
