# Data dictionary

## Source

NABirds V1 (Van Horn et al., "Building a bird recognition app and large scale dataset with citizen scientists", CVPR 2015), assembled by the Cornell Lab of Ornithology from photographs labeled and checked by birders. The brief links the copy hosted on Activeloop, which is the copy used here:

| Activeloop path | Photographs | What it is |
|---|---|---|
| `hub://activeloop/nabirds-dataset-train` | 23,928 | the official training split (the original release lists 23,929) |
| `hub://activeloop/nabirds-dataset-val` | 24,633 | the official test split; Activeloop calls it "val" and hosts no third split |

The brief asks for the original validation and test data combined as the test set; with only these two splits available, that combined set is the second row. NABirds is released for non-commercial research use.

## Raw files (`data/raw/nabirds/`, not tracked)

The photographs are 9.3 GB, so they are not in the repository. `download_nabirds()` in `src/bird_species/dataset.py` exports them once (about six minutes, no account needed) and the notebook calls it first.

| File | Contents |
|---|---|
| `train/NNNNN.jpg`, `test/NNNNN.jpg` | each photograph's original JPEG bytes, named by its position in the Activeloop split |
| `metadata.csv` | one row per photograph, columns below |
| `classes.csv` | one row per class, columns below |

`metadata.csv`:

| Column | Meaning |
|---|---|
| `split` | `train` or `test` (the Activeloop split it came from) |
| `index` | position in that split |
| `filename` | path under `data/raw/nabirds/` |
| `label` | node id in the full NABirds taxonomy (0 to 1010); only leaf nodes occur |
| `class_index` | the label re-indexed to 0..554 in taxonomy order, the target every model predicts |
| `box_x`, `box_y`, `box_w`, `box_h` | the bird's bounding box: left, top, width, height in pixels, as in the official `bounding_boxes.txt` |
| `height`, `width` | image size in pixels |

`classes.csv`:

| Column | Meaning |
|---|---|
| `label`, `class_index` | as above |
| `class_name` | the NABirds name, such as `Common Eider (Adult male)` |
| `species` | the name before the parentheses (`Common Eider`) |
| `variant` | the plumage inside the parentheses (`Adult male`), empty for a species with one class |

The 555 classes carry 404 distinct species names: 267 species have one class and 137 are split into 2 to 5 plumage classes, which make up 288 of the 555.

## Quality findings

| Finding | Evidence | Treatment |
|---|---|---|
| No corrupt or unreadable file | every file fully decoded, not just its header verified | none needed |
| All photographs are RGB and match their stored size | audit of mode and decoded size | none needed |
| No byte-identical files | SHA-256 of every file | none needed |
| 3 photographs under 100 px on a side (2 train, 1 test) | audit | kept: genuine photographs, just small |
| 183 boxes run past the photograph's edge, by up to 221 px | box right/bottom edge against image size | clipped to the edge; these are birds filling the frame, with the box drawn to the full width from an offset start |
| A perceptual hash cannot find duplicates here | 25% of training pairs at Hamming distance 0 carry different labels | rejected as the matcher |
| 655 photographs fall into 311 groups of re-saved copies and burst frames | DINOv2 cosine similarity of at least 0.96; threshold chosen from the label-free similarity distribution, checked against training labels only (1 of 141 training pairs above it disagrees) | see the next rows |
| 28 training photographs are copies of test photographs | similarity groups that reach across the split | training copies removed, so the test set loses nothing |
| 10 training photographs sit in duplicate groups with conflicting labels | same group, different training labels | removed: one of the labels must be wrong and nothing says which |
| 129 training and 178 test photographs repeat another in their own split | similarity groups inside a split | the first of each group kept |

## Derived files (`data/processed/`, tracked, written by the notebook's last cell)

| File | Contents |
|---|---|
| `partitions.csv` | every kept photograph: `filename`, `split`, `partition` (`train`, `validation` or `test`), `class_index`, `group` (duplicate group id) |
| `removed_duplicates.csv` | every removed photograph with its `drop_reason` |
| `backbone_curves.csv` | validation accuracy, top-5 accuracy and macro F1 of the prototype classifier per backbone, number of shots and sampling seed |
| `episodic_layer.csv` | per species split: the number of novel classes and their five-shot accuracy with frozen DINOv2 features and with the layer trained on base-class episodes |
| `labeling_history.csv` | per round of the brief's labeling loop: labeled photographs, share of wrong labels (`label_noise`), validation and test accuracy, proposals, confirmations and their share |
| `labeler_robustness.csv` | per simulated labeler, mistake seed and round of the brief's loop: labeled photographs, share of wrong labels, validation and test accuracy, proposals and confirmations |
| `query_strategies.csv` | per selection strategy, seed and round of the loop in which the labeler names each photograph's class: labeled photographs and validation accuracy |
| `head_search.csv` | every Optuna trial of the DINOv2 head search with its hyperparameters and best validation accuracy |
| `finetune_history.csv` | per epoch of the fine-tuning run carried forward: training and validation loss and accuracy |
| `finetune_runs.csv` | every full DINOv2 fine-tuning run (three seeds of the recipe and four one-change variants): learning rate, epochs, unfrozen blocks, seed, best validation accuracy and test accuracy |
| `test_results.csv` | every classifier on the combined test set: labels used, accuracy with its 95% Wilson interval, top-5 accuracy, macro F1, species-level accuracy |
| `calibration.csv` | test accuracy, mean confidence, expected calibration error and negative log-likelihood of the trained head and of the fine-tuned network before and after temperature scaling |
| `lime_scores.csv` | per explained test photograph: its group (best, worst, smallest birds, largest birds), share of LIME's evidence inside the bird's box, the box's share of the input, their ratio (`lift`), and for the size groups whether the prediction was right |
| `lime_stability.csv` | per re-explained photograph and sampling seed: the overlap (intersection over union) of its evidence mask with seed 42's, and the change in the share inside the box |

## Caches (`data/interim/` and `models/`, not tracked)

Every cached result carries a fingerprint of what produced it (the photographs and labels, the starting weights, every setting and the seed): `fingerprint.txt` inside a run folder, `<file>.fingerprint` beside a cached file. A result is reused only when the fingerprint of the current request matches, so a change to any input recomputes it instead of reloading a stale result under the same name.

| Path | Contents |
|---|---|
| `data/interim/embeddings/<backbone>.npz` | frozen features of every photograph for ResNet-50, CLIP ViT-B/16 and DINOv2 ViT-B/14, with the filename order |
| `data/interim/embeddings/dinov2_vit_b14_448px.npz` | DINOv2 features of the training and validation photographs at 448 pixels |
| `data/interim/probabilities/<run>_<partition>.npy` | class probabilities of each fine-tuned run on the test set, and of the models carried forward on validation |
| `data/interim/lime/*.npz` | LIME evidence masks and scores: best and worst predictions, the size samples, and the reruns with other seeds |
| `models/head/` | the Optuna study and every DINOv2 head run's best checkpoint and training log, including the head on 448-pixel features |
| `models/head_resnet50/` | the same for the ResNet-50 head |
| `models/finetune/` | the short learning-rate checks (`short_lr_*`), the recipe's three seeds (`recipe_seed*`), the one-change variants (`epochs_8`, `blocks_12`, `lr_1e-05`, `lr_0.0001`) and the ResNet-50 runs (`resnet50_lr_*`), each with its best checkpoint and training log |
