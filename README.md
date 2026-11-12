# Bird species classification with few labels

**Module 4, Sprint 3: Image classification with deep learning** | Turing College, Data Science Program

Classify North American birds into 555 fine-grained categories in two ways. First with at most five labeled photographs per class and a labeling loop that grows the labeled set from the classifier's own most confident predictions, then with a pretrained network trained on the fully labeled training set. Compare the two on the combined test set, check how far the confidence of the trained network can be trusted, and explain its best and worst predictions with LIME.

Dataset: [NABirds](https://dl.allaboutbirds.org/nabirds) (Cornell Lab of Ornithology), the [Activeloop copy](https://datasets.activeloop.ai/docs/ml/datasets/nabirds-dataset/) the brief links: 48,561 photographs in 555 classes (404 species, 137 of them split by sex, age or plumage), with one bounding box per bird. The photographs are 9.3 GB and are not in the repository; the notebook downloads them on its first run, no account needed.

---

## Result

| Classifier, scored on the combined test set (24,455 photographs) | Labels used | Accuracy (95% interval) | Top-5 | Macro F1 |
|---|---|---|---|---|
| Zero-shot CLIP ViT-B/16, one caption per class | 0 | 44.1% (43.4 to 44.7) | 81.0% | 0.400 |
| One-shot prototypes, DINOv2 ViT-B/14 | 555 | 53.6% (53.0 to 54.2) | 82.4% | 0.502 |
| **Few-shot prototypes after the labeling loop** | 6,702 | **70.2% (69.7 to 70.8)** | 87.8% | 0.670 |
| Five random shots per class, prototypes | 2,774 | 72.2% (71.6 to 72.7) | 93.3% | 0.698 |
| Five random shots per class, episodically trained layer | 2,774 | 76.6% (76.0 to 77.1) | 96.4% | 0.745 |
| All labels, prototypes | 21,364 | 81.4% (80.9 to 81.9) | 96.0% | 0.795 |
| ResNet-50, trained head | 21,364 | 56.2% (55.5 to 56.8) | 82.9% | 0.530 |
| ResNet-50, head plus fine-tuned last stage | 21,364 | 68.5% (67.9 to 69.1) | 90.1% | 0.656 |
| All labels, trained linear head | 21,364 | 87.6% (87.2 to 88.0) | 98.0% | 0.856 |
| **All labels, head plus fine-tuned top blocks** | 21,364 | **90.7% (90.3 to 91.0)** | 98.7% | 0.891 |

Both approaches meet the brief. The loop starts from one labeled photograph per class, confirms the classifier's most confident predictions round by round, and crosses the 70% target after three rounds. The fine-tuned network reaches 90.7%; the same recipe with four instead of eight epochs scores 90.0% to 90.4% over three seeds, so the seed moves the result by about a third of a point. McNemar's test puts every step between the main rows far beyond chance (p < 0.0001). Both DINOv2 approaches classify a photograph in under 30 ms on a laptop GPU, because they run the same backbone; they differ in labels and training, not in speed.

**What the project found:**

- **The backbone matters most:** on the same five labels per class, DINOv2 prototypes reach 74.7% on validation, CLIP 52.8% and ResNet-50 33.3%. ResNet-50, the convolutional network of the sprint, stays 22 points behind DINOv2 even after both are fine-tuned.
- **What the labeler is shown matters as much as how many labels it gives:** confirming only confident predictions needed 6,702 labels for 70%, while five random labels per class (2,774) give 72.2%. When the labeler names the class instead of only confirming, asking about the most uncertain photographs reaches 83.9% on validation after eight rounds, against 78.4% for confident proposals and 72.2% for the brief's confirm-only loop.
- **Labeler mistakes compound:** a simulated labeler who slips 20% of the time, or confirms half of the look-alike proposals, leaves about a tenth of the labels wrong by round 12, and the loop stalls near 70% instead of climbing to 73.6%.
- **Training beats averaging on the same labels:** a linear head removes a third of the prototypes' errors and fine-tuning the top four blocks a further quarter. With only five labels per class, a small layer trained on practice episodes, the idea behind prototypical networks, lifts the prototypes from 72.2% to 76.6%.
- **The network is underconfident, and the mistakes are look-alikes:** its average confidence is 74.4% while it is right 90.7% of the time (expected calibration error 16.2%, from label smoothing); temperature scaling brings the error to 2.5%. The remaining errors are pairs birders argue about, such as the two chickadees, the two screech-owls and the scaups, and two of the six most confident mistakes look like label errors.
- **LIME's evidence sits on the bird, less so for small birds:** 92% of the evidence lies inside the box for the largest birds but 52% for the smallest. Read against the box's share of the input, the evidence on small birds is still the most concentrated (2.96 times chance against 1.22), and reruns with other seeds mark largely the same regions.

## How the project is built

```
download      Activeloop NABirds, original JPEG bytes written once (48,561 photographs, 555 classes)
audit         full decode of every file, SHA-256, sizes and modes; boxes checked against the frame
duplicates    perceptual hash tested and rejected (25% of its exact matches carry different labels);
              DINOv2 cosine similarity over all 1.2 billion pairs instead, threshold 0.96 chosen
              without test labels; 345 copies and burst frames removed, 28 of them train-test leaks
partitions    train 21,364 / validation 2,397 (stratified, group-aware) / test 24,455 (combined set)
embeddings    frozen ResNet-50, CLIP ViT-B/16 and DINOv2 ViT-B/14 features, cached once
few-shot      prototype classifier: class-mean embeddings compared by cosine similarity; a residual
              layer trained on few-shot episodes, tested on species it never saw
labeling      one label per class to start; each round the labeler confirms or rejects the five most
              similar predictions per class; stops at 70% test accuracy as the brief specifies;
              rerun with error-prone labelers and against uncertain, random and mixed proposals
head          linear layer on frozen DINOv2 features, Optuna search over rate, weight decay, dropout
fine-tuning   top 4 of 12 blocks unfrozen, layer-wise learning-rate decay, augmentation, best epoch
              kept; three seeds, then one-change variants (8 epochs, 12 blocks, two learning rates),
              a variant kept only if it beats the seed range; ResNet-50 through the same two stages
evaluation    Wilson intervals, McNemar, top-5, macro F1, species-level accuracy, size quartiles,
              seed spread, calibration (ECE, reliability, temperature scaling), timing
LIME          20 best and 20 worst predictions plus 20 small and 20 large birds; evidence graded
              against the bird's box and its share of the input; reruns with two more seeds
```

The validation set runs optimistic for every model built from training photographs, by about three points for DINOv2 and six to seven for ResNet-50, and the notebook traces why: not closeness to the training data and not class mix, but the split itself (zero-shot CLIP, which never saw a training photograph, scores the same on validation and test). Validation still ranked every choice correctly, as the seed runs show; the numbers to quote are the test numbers above.

Every cached result (features, trained weights, predictions, LIME masks) is stored with a fingerprint of the data, labels, starting weights, settings and seed that produced it, so a change to any of them recomputes the result instead of reusing a stale one.

## Repository structure

```
.
├── data/
│   ├── raw/nabirds/                  <- photographs and metadata, downloaded by the notebook (not tracked)
│   ├── interim/                      <- cached features, predictions, LIME masks (not tracked)
│   └── processed/                    <- partitions, removed duplicates, every results table (tracked)
├── models/                           <- head searches, checkpoints and training logs (not tracked)
├── notebooks/
│   └── bird_species_classification.ipynb
├── src/bird_species/
│   ├── config.py                     <- every constant, path, threshold and hyperparameter
│   ├── cache.py                      <- fingerprints that key every cached result
│   ├── dataset.py                    <- download, audit, box geometry, partitions, pixel statistics
│   ├── duplicates.py                 <- hash and embedding duplicate search, cleaning rules
│   ├── embeddings.py                 <- frozen backbones, cached features, CLIP captions
│   ├── fewshot.py                    <- prototype classifier, proposals, the labeling loop,
│   │                                    simulated labelers, selection strategies
│   ├── episodic.py                   <- species split, residual layer trained on few-shot episodes
│   ├── statistics.py                 <- Wilson intervals and McNemar's test (statsmodels)
│   ├── explain.py                    <- LIME wrapper, evidence-on-bird check, lift, mask overlap
│   ├── plots.py                      <- every figure, returns a Figure
│   └── modeling/
│       ├── train.py                  <- linear head, Optuna search, fine-tuning for DINOv2 and ResNet-50
│       └── evaluate.py               <- test metrics, error breakdowns, calibration, validation-test gap, timing
├── tests/                            <- 126 tests, one file per module, on synthetic data
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

Any other Python 3.10 or newer environment works too, such as a conda environment used across the course: run `pip install -e .` inside it, then pick its kernel. The device is detected with `torch.accelerator`, so the same code uses a CUDA GPU, an Apple GPU or the CPU.

Or end to end from the command line:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/bird_species_classification.ipynb
```

A first run downloads the photographs (about six minutes), extracts features with three backbones (10 to 20 minutes each on an Apple M4, plus about 70 minutes for the 448-pixel features), searches both heads (a few minutes), and trains ten full fine-tuning runs (about an hour each for DINOv2, two hours for eight epochs or twelve unfrozen blocks, 20 minutes each for ResNet-50), then computes the LIME explanations, roughly 13 hours in all. Every one of those results is cached, so later runs repeat the whole analysis in about five minutes. The Mac must stay awake while it trains: a closed lid on battery suspends the run.

Checks:

```bash
uv run pytest -q        # 126 tests
uv run ruff check .     # package, tests and notebook, line length included
```

## What to improve next

- Let the labeler name the class and propose the uncertain photographs, and audit a sample of each round's confirmations, starting with the look-alike pairs.
- Give small birds more pixels by classifying a detector's crop of the bird; frozen 448-pixel features already help the five-shot prototypes on the smallest birds.
- Use the calibrated probabilities as a reject option: answer with the species or the look-alike pair, or ask a birder, when the confidence falls below a threshold.
- Have an expert recheck NABirds' age classes, starting with the most confident mistakes.
- Hold validation out by photographer or date if NABirds' original metadata is available, so validation predicts test.
