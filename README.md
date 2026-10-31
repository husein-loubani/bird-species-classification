# Bird species classification with few labels

**Module 4, Sprint 3: Image classification with deep learning** | Turing College, Data Science Program

Classify North American birds into 555 fine-grained categories in two ways. First with at most five labeled photographs per class and a labeling loop that grows the labeled set from the classifier's own most confident predictions, then with a pretrained network trained on the fully labeled training set. Compare the two on the combined test set and explain the trained network's best and worst predictions with LIME.

Dataset: [NABirds](https://dl.allaboutbirds.org/nabirds) (Cornell Lab of Ornithology), the [Activeloop copy](https://datasets.activeloop.ai/docs/ml/datasets/nabirds-dataset/) the brief links: 48,561 photographs in 555 classes (404 species, 137 of them split by sex, age or plumage), with one bounding box per bird. The photographs are 9.3 GB and are not in the repository; the notebook downloads them on its first run, no account needed.

---

## Result

| Classifier, scored on the combined test set (24,455 photographs) | Labels used | Accuracy (95% interval) | Top-5 | Macro F1 |
|---|---|---|---|---|
| Zero-shot CLIP ViT-B/16, one caption per class | 0 | 44.1% (43.4 to 44.7) | 81.0% | 0.400 |
| One-shot prototypes, DINOv2 ViT-B/14 | 555 | 53.6% (53.0 to 54.2) | 82.4% | 0.502 |
| **Few-shot prototypes after the labeling loop** | 6,702 | **70.2% (69.7 to 70.8)** | 87.8% | 0.670 |
| Five random shots per class, prototypes | 2,774 | 72.2% (71.6 to 72.7) | 93.3% | 0.698 |
| All labels, prototypes | 21,364 | 81.4% (80.9 to 81.9) | 96.0% | 0.795 |
| All labels, trained linear head | 21,364 | 87.6% (87.2 to 88.0) | 98.0% | 0.856 |
| **All labels, head plus fine-tuned top blocks** | 21,364 | **90.0% (89.6 to 90.4)** | 98.6% | 0.883 |

Both approaches meet the brief. The loop starts from one labeled photograph per class, confirms the classifier's most confident predictions round by round, and crosses the 70% target after three rounds. The fine-tuned network reaches 90.0%, and McNemar's test puts every step between the main rows far beyond chance (p < 0.0001). Both classify a photograph in under 30 ms on a laptop GPU, because they run the same backbone; they differ in labels and training, not in speed.

**Three findings decide the story:**

- **The backbone matters most:** on the same five labels per class, DINOv2 prototypes reach 74.7% on validation, CLIP 52.8% and ResNet-50 33.3%. A single DINOv2 shot (54.2%) beats CLIP's zero-shot captions (44.1%).
- **Confirming only confident predictions is label-hungry:** the loop needed 6,702 labels for 70%, while five random labels per class (2,774) give 72.2%: a confident prediction resembles what is already labeled, so it adds little. Ranking proposals by similarity instead of the usual margin still saved a third of the labels (6,702 against 10,153 on validation).
- **Training beats averaging on the same labels:** a linear head removes a third of the prototypes' errors, and fine-tuning the top four blocks removes a further fifth. The remaining errors are classic look-alikes: Greater and Lesser Scaup, the two screech-owls, Pacific and Winter Wren. LIME puts a median of 92% to 94% of its evidence inside the bird's box for both the best and the worst predictions, so the mistakes come from misreading the bird, not from the background.

## How the project is built

```
download      Activeloop NABirds, original JPEG bytes written once (48,561 photographs, 555 classes)
audit         full decode of every file, SHA-256, sizes and modes; boxes checked against the frame
duplicates    perceptual hash tested and rejected (25% of its exact matches carry different labels);
              DINOv2 cosine similarity over all 1.2 billion pairs instead, threshold 0.96 chosen
              without test labels; 345 copies and burst frames removed, 28 of them train-test leaks
partitions    train 21,364 / validation 2,397 (stratified, group-aware) / test 24,455 (combined set)
embeddings    frozen ResNet-50, CLIP ViT-B/16 and DINOv2 ViT-B/14 features, cached once
few-shot      prototype classifier: class-mean embeddings compared by cosine similarity
labeling      one label per class to start; each round the labeler confirms or rejects the five most
              similar predictions per class; stops at 70% test accuracy as the brief specifies
head          linear layer on frozen DINOv2 features, Optuna search over rate, weight decay, dropout
fine-tuning   top 4 of 12 blocks unfrozen, layer-wise learning-rate decay, augmentation, best epoch kept
evaluation    Wilson intervals, McNemar, top-5, macro F1, species-level accuracy, size quartiles, timing
LIME          20 best and 20 worst predictions explained; evidence graded against the bird's box
```

The validation set runs about three points optimistic for every model built from training photographs, and the notebook traces why: not closeness to the training data and not class mix, but the split itself (zero-shot CLIP, which never saw a training photograph, scores the same on validation and test). Choices made on validation stay valid; the numbers to quote are the test numbers above.

## Repository structure

```
.
├── data/
│   ├── raw/nabirds/                  <- photographs and metadata, downloaded by the notebook (not tracked)
│   ├── interim/                      <- cached features, predictions, LIME masks (not tracked)
│   └── processed/                    <- partitions, removed duplicates, every results table (tracked)
├── models/                           <- head search, checkpoints and training logs (not tracked)
├── notebooks/
│   └── bird_species_classification.ipynb
├── src/bird_species/
│   ├── config.py                     <- every constant, path, threshold and hyperparameter
│   ├── dataset.py                    <- download, audit, box geometry, partitions, pixel statistics
│   ├── duplicates.py                 <- hash and embedding duplicate search, cleaning rules
│   ├── embeddings.py                 <- frozen backbones, cached features, CLIP captions
│   ├── fewshot.py                    <- prototype classifier, proposals, the labeling loop
│   ├── statistics.py                 <- Wilson intervals and McNemar's test (statsmodels)
│   ├── explain.py                    <- LIME wrapper and the evidence-on-bird check
│   ├── plots.py                      <- every figure, returns a Figure
│   └── modeling/
│       ├── train.py                  <- linear head, Optuna search, fine-tuning with layer-wise decay
│       └── evaluate.py               <- test metrics, error breakdowns, validation-test gap, timing
├── tests/                            <- 95 tests, one file per module, on synthetic data
├── references/data_dictionary.md
├── reports/figures/
├── pyproject.toml
└── uv.lock
```

## How to run

```bash
uv sync --extra dev
uv run jupyter lab notebooks/bird_species_classification.ipynb    # kernel: "Python 3 (ipykernel)"
```

Any other Python 3.10 or newer environment works too, such as a conda environment used across the course: run `pip install -e .` inside it, then pick its kernel. The cached results below make that run take minutes.

Or end to end from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/bird_species_classification.ipynb
```

A first run downloads the photographs (about six minutes), extracts features with three backbones (10 to 20 minutes each on an Apple M4), searches the head (a few minutes) and fine-tunes the network (about an hour). Every one of those results is cached, so later runs repeat the whole analysis in about three minutes. The Mac must stay awake while it trains: a closed lid on battery suspends the run.

Checks:

```bash
uv run pytest -q        # 95 tests
uv run ruff check .     # package, tests and notebook
```

## What to improve next

- Propose a mix of confident and uncertain photographs in the labeling loop, and let the labeler correct wrong proposals instead of only rejecting them; random labeling already beats confident-only labeling here.
- Give small birds more pixels: fine-tune at 448 pixels, or classify a detector's crop of the bird. Both models are weakest when the bird fills under a sixth of the frame.
- Fine-tune more of the network with several seeds, to put an interval on the 90.0% and test whether unfreezing more blocks helps.
- Answer at the species level, or name the look-alike pair, when the top two classes are plumages of one species or a known confusion.
- Hold validation out by photographer or date if NABirds' original metadata is available, so validation predicts test.
