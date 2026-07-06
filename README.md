# crispAIPE: Probabilistic Modelling of Prime Editing Variant Correction Efficiency

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Training the Model](#training-the-model)
- [Testing and Inference](#testing-and-inference)
- [Ablation Studies](#ablation-studies)
- [Configuration Options](#configuration-options)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- PyTorch Lightning
- CUDA (optional, for GPU acceleration)
- Apple Silicon MPS support (for M1/M2 Macs)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pe-uncert
```

2. Install the package:
```bash
pip install -e .
```

3. Install additional dependencies:
```bash
pip install pytorch-lightning wandb scipy scikit-learn pandas numpy
```

4. (Optional) Activate your virtual environment:
```bash
source /path/to/your/venv/bin/activate
# you may use pe-uncert.yaml env provided.
```

## Quick Start

### Training a Model

For the canonical target-disjoint training run (matches the published numbers):

```bash
./train_crispAIPE_revs.sh
```

This wraps the `crispAIPE_revs_conf.json` config and points at the
`pridict-{train,val,test}_revs.csv` data files. To override hyperparameters
inline:

```bash
./train_crispAIPE_revs.sh \
  --batch_size 128 \
  --lr 0.0006 \
  --epochs 100
```

`train_crispAIPE.sh` (without `_revs`) is the legacy random-split wrapper, kept
for audit purposes only.

### Testing with Random Samples

Test the model with random batches and single examples after training a target-
disjoint checkpoint:

```bash
python3 test_crispAIPE_samples.py \
  --config pe_uncert_models/configs/crispAIPE_revs_conf.json \
  --checkpoint pe_uncert_models/logs/crispAIPE_revs_conf/<timestamp>/best_model-epoch=<epoch>-val_loss_val_loss=<loss>.ckpt \
  --batch_size 16
```

## Data Requirements

The model expects CSV files with the following columns:

- `initial_sequence`: The original DNA sequence (one-hot encoded)
- `mutated_sequence`: The target mutated sequence (one-hot encoded)
- `total_read_count`: Total number of sequencing reads
- `edited_percentage`: Percentage of edited reads
- `unedited_percentage`: Percentage of unedited reads
- `indel_percentage`: Percentage of indel reads
- `protospacer_location`: Binary mask indicating protospacer location
- `pbs_location`: Binary mask indicating PBS location
- `rt_initial_location`: Binary mask indicating RT template initial location
- `rt_mutated_location`: Binary mask indicating RT template mutated location

### Data Format

The model uses the PRIDICT dataset format. Canonical data files (target-disjoint split):
- Training:   `data/pridict_data_revs/pridict-train_revs.csv` (64,751 pegRNAs, 9,344 mutation groups)
- Validation: `data/pridict_data_revs/pridict-val_revs.csv`   (13,828 pegRNAs, 2,002 mutation groups)
- Test:       `data/pridict_data_revs/pridict-test_revs.csv`  (13,844 pegRNAs, 2,003 mutation groups)

Train ↔ val ↔ test mutation groups are **disjoint by construction** (`zero overlap`).

> **Note**: the `data/` directory is gitignored. The PRIDICT Library-1 source
> spreadsheet is available as Supplementary Table 5 of
> [Mathis *et al.* 2023](https://doi.org/10.1038/s41587-022-01613-7). The
> regression variant uses the HEK293T PE2 split from
> [DeepPrime (Yu *et al.* 2023)](https://doi.org/10.1016/j.cell.2023.03.034).
> To regenerate the target-disjoint split from the source spreadsheet only,
> download Supplementary Table 5 from Mathis et al. as
> `data/pridict_data_revs/41587_2022_1613_MOESM5_ESM.xlsx`, then run:
> ```bash
> python data/pridict_data_revs/target_disjoint_split.py \
>     --excel data/pridict_data_revs/41587_2022_1613_MOESM5_ESM.xlsx \
>     --output_dir data/pridict_data_revs/ \
>     --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 --seed 42
> ```
> If a pre-cleaned `pridict-90k-cleaned.csv` is already present, it can be
> supplied with `--csv`; otherwise the script rebuilds the crispAIPE-format
> Library-1 table from the Excel PE2 columns before splitting.
> Expected output counts are 64,751 / 13,828 / 13,844 train/validation/test
> samples with zero mutation-`Name` overlap. Some protospacer overlap remains
> across partitions because the split is deliberately mutation-disjoint rather
> than protospacer-disjoint.
> See `data/pridict_data_revs/split_cmd_revs.txt` for context and the
> pre-fix random-split command (kept for audit purposes).

## Training the Model

### Using the Shell Script (Recommended)

Use the `_revs` launcher for the manuscript target-disjoint model:

```bash
./train_crispAIPE_revs.sh [OPTIONS]
```

> The wrapper will auto-source `~/aidev/bin/activate` if it exists; otherwise it
> falls through silently and uses whatever `python3`/`python` is on `$PATH`.
> Edit the path near the top of the script to point at your own virtualenv if
> you want auto-activation.

#### Basic Usage

```bash
# Train with default parameters
./train_crispAIPE_revs.sh

# Train with custom learning rate and batch size
./train_crispAIPE_revs.sh --lr 0.001 --batch_size 64

# Train with custom data paths
./train_crispAIPE_revs.sh \
  --train_data_path data/my_train.csv \
  --test_data_path data/my_test.csv
```

#### Full Example

```bash
./train_crispAIPE_revs.sh \
  --train_data_path data/pridict_data_revs/pridict-train_revs.csv \
  --val_data_path data/pridict_data_revs/pridict-val_revs.csv \
  --test_data_path data/pridict_data_revs/pridict-test_revs.csv \
  --batch_size 128 \
  --lr 0.0006 \
  --epochs 100 \
  --dropout 0.1 \
  --embedding_dim 64 \
  --early_stopping true \
  --patience 8 \
  --project_name crispAIPE_revs_conf
```

#### View All Options

```bash
./train_crispAIPE_revs.sh --help
```

### Using Python Directly

You can also train using the Python training script directly:

```bash
python pe_uncert_models/models/train.py \
  --config pe_uncert_models/configs/crispAIPE_revs_conf.json
```

### Training Output

Training logs and checkpoints are saved to:
```
pe_uncert_models/logs/<project_name>/<timestamp>/
```

The directory contains:
- `best_model-epoch=XX-val_loss=YY.ckpt`: Best model checkpoint
- `last.ckpt`: Last epoch checkpoint
- CSV/WandB logs, depending on the logger configured in the training script

## Testing and Inference

### Test with Random Samples

The `test_crispAIPE_samples.py` script allows you to test the model with random batches and single examples:

```bash
python3 test_crispAIPE_samples.py \
  --config <config_path> \
  --checkpoint <checkpoint_path> \
  --batch_size 16 \
  --seed 42 \
  --output_dir test_samples_output
```

#### Arguments

- `--config`: Path to the config JSON file used for training
- `--checkpoint`: Path to the model checkpoint file (.ckpt)
- `--batch_size`: Size of random batch to sample (default: 16)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Output directory for results (default: test_samples_output)

#### Output

The script generates:
1. **Console output**: Detailed inference results showing:
   - Loss values
   - Ground truth vs predicted proportions
   - Error metrics
   - Dirichlet alpha parameters

2. **JSON files**:
   - `batch_results.json`: Full batch inference results
   - `single_example_results.json`: Single example inference results

### Example Output

```
BATCH INFERENCE RESULTS
================================================================================
Loss: -1.462086
Batch Size: 16

Ground Truth vs Predictions (first 5 examples):
Idx   GT Edited    GT Unedited    GT Indel     Pred Edited    Pred Unedited    Pred Indel     Error     
----------------------------------------------------------------------------------------------------
0     0.1882       0.6921         0.1197       0.5607         0.1235           0.3158         0.3791    
...

Mean Errors Across Batch:
  Edited:    0.1476
  Unedited:  0.1549
  Indel:     0.0499
  Overall:   0.1175
```

## Ablation Studies

Two ablation studies are reproduced in this repository.

### Architecture ablation (Hybrid vs. Transformer-only vs. CNN-only)

Train the two ablation variants and run the joint evaluation:

```bash
# Train transformer-only variant
python pe_uncert_models/models/train_ablation.py \
    --config pe_uncert_models/configs/crispAIPE_transformer_only_revs_conf.json \
    --model_type transformer_only

# Train CNN-only variant
python pe_uncert_models/models/train_ablation.py \
    --config pe_uncert_models/configs/crispAIPE_cnn_only_revs_conf.json \
    --model_type cnn_only

# Evaluate all three architectures together
python test/model_ablation.py \
    --config pe_uncert_models/configs/crispAIPE_revs_conf.json \
    --hybrid_checkpoint <hybrid.ckpt> \
    --transformer_checkpoint <transformer.ckpt> \
    --cnn_checkpoint <cnn.ckpt> \
    --output_dir results/ablation/architecture
```


### Distribution ablation (Dirichlet vs. Softmax vs. Logit-Normal)

```bash
# Train each distributional head
python pe_uncert_models/models/train_distribution_ablation.py \
    --config pe_uncert_models/configs/crispAIPE_softmax_revs_conf.json \
    --distribution softmax

python pe_uncert_models/models/train_distribution_ablation.py \
    --config pe_uncert_models/configs/crispAIPE_logit_normal_revs_conf.json \
    --distribution logit_normal

# Evaluate side-by-side
python test/distribution_ablation.py \
    --config pe_uncert_models/configs/crispAIPE_revs_conf.json \
    --dirichlet_checkpoint <dirichlet.ckpt> \
    --softmax_checkpoint <softmax.ckpt> \
    --logit_normal_checkpoint <logit_normal.ckpt> \
    --output_dir results/ablation/distribution
```

## Configuration Options

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `target_dna_flank_len` | Target DNA flank length | 0 |
| `kmer_size` | Legacy config field; not used by the active crispAIPE forward path | 3 |
| `n_embd` | Embedding dimension | 64 |
| `d_model` | Model dimension | 16 |
| `n_layer` | Number of transformer layers | 4 |
| `dropout` | Dropout rate | 0.1 |
| `embedding_dim` | Embedding dimension | 64 |
| `nhead` | Number of attention heads | 4 |
| `bottleneck_dim` | Bottleneck dimension | 8 |
| `assesor_type` | Assessor type (multinomial/softmax/logit_normal) | multinomial |

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `train_data_path` | Path to training CSV | `data/pridict_data_revs/pridict-train_revs.csv` |
| `val_data_path` | Path to validation CSV | `data/pridict_data_revs/pridict-val_revs.csv` |
| `test_data_path` | Path to test CSV | `data/pridict_data_revs/pridict-test_revs.csv` |
| `batch_size` | Batch size | 128 |
| `val_split` | Validation split ratio | 0.1 |
| `sequence_length` | Sequence length | 99 |
| `pegrna_length` | PEGRNA length | 99 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate | 6e-4 |
| `weight_decay` | Weight decay | 0.01 |
| `warmup_epochs` | Warmup epochs | 10 |
| `max_epochs` | Maximum epochs | 100 |
| `early_stopping` | Enable early stopping | true |
| `patience` | Early stopping patience | 8 |
| `cpu` | Use CPU only | false |
| `gpus` | GPU IDs (comma-separated) | "" (auto-detect) |

## Project Structure

```
pe-uncert/
├── pe_uncert_models/                       # Main package
│   ├── models/                             # Model definitions
│   │   ├── crispAIPE.py                    # Hybrid Transformer+CNN with Dirichlet head
│   │   ├── crispAIPE_regression.py         # MSE regression variant (crispAIPE-reg)
│   │   ├── distribution_ablation.py        # Softmax / Logit-Normal heads
│   │   ├── train.py                        # Main training entry point
│   │   ├── train_regression.py             # Regression training entry point
│   │   ├── train_ablation.py               # Architecture-ablation training
│   │   └── train_distribution_ablation.py  # Distribution-ablation training
│   ├── data_utils/                         # Data loading and grouped splitting
│   ├── configs/                            # Config JSONs (see below)
│   └── vocab/                              # legacy vocabulary files
├── data/                                   # PRIDICT / DeepPrime CSVs (gitignored)
├── test/                                   # Evaluation, figures, ablations
│   ├── diagnostics_plot*.py                # Calibration / diagnostics plots
│   ├── attention_*.py                      # Attention-based interpretability
│   ├── spearman_correlations.py            # Outcome-wise correlation analysis
│   ├── model_ablation.py                   # Architecture-ablation evaluation
│   ├── distribution_ablation.py            # Distribution-ablation evaluation
│   ├── run_ablation_study.py               # End-to-end ablation automation
│   └── figures/                            # Generated figures and tables
├── manuscript/                             # OUP/Bioinformatics LaTeX source
├── train_crispAIPE_revs.sh                 # Target-disjoint training launcher
├── train_crispAIPE.sh                      # Legacy random-split training launcher
├── test_crispAIPE_samples.py               # Sample-level inference script
└── README.md                               # This file
```

## Configuration Files

Pre-configured settings are available in `pe_uncert_models/configs/`. The `_revs`
suffix indicates configs that point at the target-disjoint split files
(`pridict-{train,val,test}_revs.csv`); the unsuffixed variants point at the
older random split and are kept for reproducibility audits only.

- `crispAIPE_revs_conf.json` — **Main Dirichlet model on target-disjoint split (recommended)**
- `crispAIPE_softmax_revs_conf.json` — Softmax distribution-ablation on target-disjoint split
- `crispAIPE_logit_normal_revs_conf.json` — Logit-Normal distribution-ablation on target-disjoint split
- `crispAIPE_cnn_only_revs_conf.json` — CNN-only architecture ablation on target-disjoint split
- `crispAIPE_transformer_only_revs_conf.json` — Transformer-only architecture ablation on target-disjoint split
- `crispAIPE_regression_deepprime_conf.json` — `crispAIPE-reg` (MSE regression) on the DeepPrime HEK293T PE2 split
- `crispAIPE_train_test_split_conf.json`, `crispAIPE_conf1.json`, and the four non-`_revs` ablation configs — *legacy random-split configs; do not use for the published numbers.*

## Advanced Usage

### Custom Model Variants

The codebase includes several model variants:

1. **crispAIPE (Default)** — Hybrid Transformer + CNN with Dirichlet output (`pe_uncert_models/models/crispAIPE.py`)
2. **crispAIPE-reg** — MSE-regression efficiency-only variant (`crispAIPE_regression.py`, train via `train_regression.py`)
3. **crispAIPE_Softmax / crispAIPE_LogitNormal** — Distribution-ablation heads (`distribution_ablation.py`)
4. **TransformerOnly / CNNOnly** — Architecture-ablation variants (used by `train_ablation.py`)

### Using Different Configurations

Use one of the provided config files (the wrapper sets `assesor_type` from the
config; pass it explicitly only when overriding):

```bash
./train_crispAIPE_revs.sh --assesor_type softmax
```

### Regression variant (`crispAIPE-reg`)

```bash
python pe_uncert_models/models/train_regression.py \
    --config pe_uncert_models/configs/crispAIPE_regression_deepprime_conf.json
```

The regression results figures used in the manuscript are produced by
`test/plot_regression_results.py` and `test/regression_performance_comparison.py`.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or use CPU mode (`--cpu true`)
2. **File Not Found**: Ensure data paths are correct relative to the config file location
3. **Import Errors**: Make sure the package is installed (`pip install -e .`)

### Device Selection

The script automatically detects and uses:
- CUDA GPUs (if available)
- Apple Silicon MPS (M1/M2 Macs)
- CPU (fallback)

Force CPU mode:
```bash
./train_crispAIPE.sh --cpu true
```

## Citation

If you use this code in your research, please cite:

```bibtex
@unpublished{ozden2026crispaipe,
  title  = {Probabilistic Modelling of Prime Editing Variant Correction Efficiency},
  author = {{\"O}zden, Furkan and Lu, Peiheng and Minary, Peter},
  year   = {2026},
  note   = {Preprint; under submission}
}
```

## License

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

This work is funded by Google DeepMind.
The authors have applied a **CC BY** public copyright licence to any Author
Accepted Manuscript (AAM) version arising from this submission.

## Contact

Furkan Özden — `furkan.ozden@cs.ox.ac.uk`
Peter Minary (corresponding author) — `peter.minary@cs.ox.ac.uk`
Department of Computer Science, University of Oxford.
