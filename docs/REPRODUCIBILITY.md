# Reproducibility notes

## Scope

This repository provides a checkpoint-compatible implementation of DI-NIDS and the supplied
feature-extractor and source-classifier state dictionaries. The complete numerical tables from the
paper have not yet been regenerated in the packaged environment. Exact dataset snapshots and the
original execution environment are not distributed here, so new results should record the items
listed under [Reporting a run](#reporting-a-run).

## Data representation

The implementation expects the 43 NetFlow-v2 fields used in the paper, together with the `Attack`
and binary `Label` columns. Four flow identifiers are removed before training, leaving 39 model
inputs. The field list is defined in `src/dinids/constants.py`.

## Implemented architecture

The supplied checkpoints require the following layer layout:

| Component | Checkpoint-compatible layout |
| --- | --- |
| Feature extractor | `39→10→10→10→10`, with ReLU after each linear layer and dropout after the middle two activations |
| Label classifier | `10→10→1`, with a ReLU hidden activation and one binary logit |
| Domain classifier | Gradient reversal followed by `10→10→10→10→1`, with ReLU and dropout and one binary logit |

Table 1 of the paper presents the DANN architecture at a higher level. The layout above is the exact
implementation used here and is required by the shapes of the supplied state dictionaries. A
one-logit sigmoid classifier and a two-logit softmax classifier can both model a binary outcome,
but their checkpoint shapes are not interchangeable.

## Preprocessing modes

Two preprocessing modes are available because scaling materially affects cross-domain results:

| Mode | Behaviour | Comparison guidance |
| --- | --- | --- |
| `legacy-independent` | Fits a separate min-max scaler to each complete domain before splitting | Use with the supplied checkpoints and when seeking compatibility with the original experiments |
| `source-train` | Fits one scaler to the source training split and applies it to all splits | Treat as a new methodology and do not compare directly with legacy-mode results |

`legacy-independent` is transductive because it uses feature ranges from the complete source and
target domains. In this mode, target `Attack` metadata is used only to stratify the held-out split;
target labels are never passed to the source-classification or domain-adversarial loss.

## Supplied checkpoints

The checkpoint files are PyTorch state dictionaries supplied with the research code and renamed by
source domain for clarity. Their SHA-256 hashes are recorded in `models/SHA256SUMS`. The supplied
files include two feature extractors and two source classifiers. They do not include trained domain
classifiers, fitted OSVMs, scaler objects, dataset checksums, or training metadata.

Use `encoder_source_cic2018.pt` when CIC-2018 is the source domain and
`encoder_source_unsw_nb15.pt` when UNSW-NB15 is the source domain.

## Metric definitions

Scikit-learn's One-Class SVM predicts `+1` for inliers, which are benign flows here, and `-1` for
outliers, which are attacks. Evaluation output therefore names benign-positive and attack-positive
precision, recall, and F1 separately. It also reports macro-F1, accuracy, attack ROC-AUC from the
continuous anomaly score, and a labelled confusion matrix.

These explicit definitions may produce values that differ from scripts that use hard predictions
for ROC-AUC or leave the positive class unspecified.

## Reporting a run

For a result that others can compare or reproduce, record:

1. Dataset names, versions, download sources, and SHA-256 hashes.
2. Source and target direction.
3. Git commit or release tag.
4. Python, PyTorch, CUDA, scikit-learn, and hardware versions.
5. Complete command-line invocation and random seed.
6. Preprocessing and optimisation modes, epoch count, batch size, and any row caps.
7. Checkpoint hashes and the generated JSON result file.

Full-dataset reproduction of both cross-domain directions remains necessary before claiming exact
numerical agreement with the published tables.
