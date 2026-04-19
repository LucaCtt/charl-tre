# CHARL-TRE

Activity recognition from Wi-Fi Channel State Information (CSI) using a categorical VAE with Gumbel-Softmax latents, followed by causal discovery and deterministic symbolic rule-based classification.

The pipeline has two stages:

1. **VAE training** — learn a discrete latent representation of CSI windows via Gumbel-Softmax, with Optuna hyperparameter search over multiple GPUs.
2. **Causal analysis** — run LPCMCI on the per-activity latent time-series, extract discriminative causal paths, mine symbolic IF-THEN rules, calibrate and evaluate a deterministic classifier.

## Requirements

- Python 3.13
- CUDA-capable GPU(s) for training (CPU fallback works but is slow)
- [`uv`](https://github.com/astral-sh/uv) for dependency management

## Installation

```bash
git clone <repo-url>
cd charl_tre
uv sync
```

## Dataset

The code expects the [Exposing the CSI - S1 dataset](https://zenodo.org/records/7732595) in MATLAB `.mat` format, one file per activity, named `S1a_A.mat` through `S1a_L.mat` (one letter per activity, in the order defined by `settings.activities`).

Place files at the path configured by `dataset_path` (default: `dataset/S1/`):

```
dataset/S1/
  S1a_A.mat   # Walk
  S1a_B.mat   # Run
  S1a_C.mat   # Jump
  ...
  S1a_L.mat   # Stretch
```

## Configuration

All settings are managed via `charl_tre.settings.Settings`, a Pydantic `BaseSettings` class. Defaults are in the class definition; any field can be overridden with an environment variable or a `.env` file at the project root.

## Usage

### Stage 1 — VAE Hyperparameter Search

```bash
uv run opt
```

Launches a multi-GPU Optuna study (one process per GPU via `torch.multiprocessing.spawn`). Each trial trains a `SingleAntennaVAE` with a different hyperparameter combination. Trials are pruned early by `CollapsePruner`, which detects irreversible categorical posterior collapse by monitoring latent entropy over the last `patience` epochs.

Outputs are written to `out/<study_name>/`:

```
out/<study_name>/
  trial_0/
    model.pt
    results.json
  trial_1/
    ...
  study_results.json    # best trial number and value
```

### Stage 1b — Neural Classifier Evaluation

```bash
uv run test
```

Loads the best VAE checkpoint from `study_results.json`, trains a small MLP classifier on the frozen latent representations (using the larger `test_window_size`), then evaluates on the held-out test set. Saves `confusion_matrix.png` and `latent_tsne.png` to the study output directory.

### Stage 1c — Latent Extraction

Open and run `notebooks/compute_latents.ipynb`. This encodes the full dataset with the best VAE and saves:

```
out/<study_name>/latents/
  latents_hard.npy    # (T, latent_dim, n_categories) one-hot hard samples
  latents_soft.npy    # (T, latent_dim, n_categories) soft logits
  labels.npy          # (T,) integer activity labels
```

### Stage 2 — Causal Analysis

Open and run `notebooks/causal_analysis.ipynb`. The notebook is split into two independent stages:

**Stage 2a — Raw causal discovery**

```python
RawDiscoveryPipeline(settings=settings, ind_test_name="parcorr", ...).run()
```

Runs LPCMCI (from [Tigramite](https://github.com/jakobrunge/tigramite)) independently for each activity on its latent time-series. Results are saved as JSON to `out/<study_name>/causal_graphs_raw/<ind_test>/`.

**Stage 2b — Path search and rule mining**

```python
PathSearchPipeline(settings=settings, raw_dir=raw_dir, ...).run()
```

Loads the raw graphs, extracts directed causal edges, enumerates discriminative paths via DFS, mines symbolic IF-THEN rules from the edges, calibrates and evaluates a deterministic classifier. Outputs go to `out/<study_name>/causal_path_search/<ind_test>/`:

```
out/<study_name>/causal_path_search/parcorr/
  classification_rules.json
  classification_rules.txt
  path_library.json
  graphs_summary.json
  deterministic_classifier_metrics.json
  deterministic_confusion_matrix.png
  <activity_name>/
    graph.npy
    val_matrix.npy
    edges.json
    edges.csv
    <activity_name>.pdf    # causal graph figure
```

## Model Architecture

The VAE encoder maps a `(1, window_size, n_subcarriers)` CSI spectrogram through two strided Conv2d layers to a flat feature vector, then projects to `latent_dim × n_categories` logits. A straight-through Gumbel-Softmax produces a one-hot hard sample `z_hard` of shape `(batch, latent_dim, n_categories)`. The decoder mirrors this with transposed convolutions back to the original input shape.

The loss is MSE reconstruction plus a capacity-controlled categorical KL divergence:

```
L = MSE(x_recon, x) + kl_weight * max(0, KL(q || Uniform) - capacity)
```

Annealing schedules are applied to `kl_weight`, `capacity`, and the Gumbel temperature `tau` over training epochs.

## Causal Analysis Details

LPCMCI is a constraint-based causal discovery algorithm for time series with latent confounders. It is run separately for each activity, treating each one-hot latent variable `Z{i}_C{j}` as a node in the time-series graph.

The causal pipeline components:

- **`LPCMCIRunner`** — wraps Tigramite, supports `parcorr`, `robustparcorr`, `gpdc`, `gsquared`, `cmisymb`, `cmiknn`. Runs activities in parallel via `ThreadPoolExecutor`. For G-squared tests, infers link signs from Spearman correlation.
- **`GraphProcessor`** — filters weak/ambiguous edges, enumerates all DFS paths up to `max_edges_per_path` edges, scores paths by cumulative edge strength.
- **`PathSearcher`** — assigns one unique path per activity maximising discriminative margin via the Hungarian algorithm.
- **`RuleBuilder`** — converts edges to `IF src[t-lag] is active THEN dst[t] is likely {true|false}` rules. Uses log-linear discriminativity weighting (`delta × log(count)`), greedy diversity selection, and a fallback miner for activities with sparse graphs.
- **`ActivityScorer`** — scores sequences via log-likelihood ratio against per-class rules, adds a prototype similarity term for sparse-edge activities, and calibrates score biases on held-out train segments.

## License

MIT. See `LICENSE`.

## Acknowledgements

The work was partially supported by the European Office of Aerospace Research & Development under award numbers FA8655-22-1-7017 and FA8655-25-1-7067, and by the U.S. Army DEVCOM Army Research Laboratory (ARL) under Cooperative Agreement #W911NF2220243. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the authors or of the United States government.

## Author

Luca Cotti (<luca.cotti@unibs.it>)