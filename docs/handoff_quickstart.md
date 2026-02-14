# Handoff Quickstart (New Computer)

This is the minimal runbook to clone, set up, and run the implemented pipeline.

## 1) One-time setup on a new machine

```bash
git clone https://github.com/AndrewfDanford/Masters.git
cd Masters

bash scripts/setup_env.sh
```

If `torch` install fails on macOS, use the PyTorch selector:
- <https://pytorch.org/get-started/locally/>

Quick smoke check (no data needed):

```bash
bash scripts/run_faithfulness_demo.sh
```

Optional monitoring dashboard:

```bash
bash scripts/run_dashboard.sh
```

From the dashboard sidebar, use `Run Synthetic Smoke` for a one-click data-free integration run.

## 2) Run when MIMIC data is available

### One command path (recommended)

```bash
export MIMIC_METADATA_CSV=/absolute/path/to/mimic-cxr-2.0.0-metadata.csv.gz
export MIMIC_LABELS_CSV=/absolute/path/to/mimic-cxr-2.0.0-chexpert.csv.gz
export MIMIC_OFFICIAL_SPLIT_CSV=/absolute/path/to/mimic-cxr-2.0.0-split.csv.gz
export E1_IMAGE_ROOT=/absolute/path/to/mimic-cxr-jpg

# Recommended on Apple Silicon if available:
export E1_CNN_DEVICE=mps
export E23_DEVICE=mps

# Optional first pass for speed:
# export E1_CNN_MAX_SAMPLES_PER_SPLIT=300
# export E23_MAX_SAMPLES=200

# Optional: text-family proxies are disabled by default in core scope.
# To enable future-work text tracks explicitly:
# export E5_RUN=1
# export E6_RUN=1

bash scripts/run_after_data.sh
```

## 3) Main outputs to check

- `outputs/models/e1_cnn_checkpoint.pt`
- `outputs/reports/e1_cnn/e1_cnn_metrics_summary.csv`
- `outputs/reports/e2_e3/e2e3_saliency_method_summary.csv`
- `outputs/reports/e4/e4_artifacts.csv` (proxy unless replaced by model-backed E4)
- `outputs/reports/e5/e5_artifacts.csv` (optional future-work path; only if enabled)
- `outputs/reports/e6/e6_artifacts.csv` (optional future-work path; only if enabled)
- `outputs/reports/e7/e7_input_all_methods.csv`
- `outputs/reports/e7/e7_method_summary.csv`
- `outputs/reports/e8/e8_randomization_method_summary.csv`

## 4) Synthetic smoke (no private data required)

```bash
bash scripts/run_synthetic_smoke.sh
```

Key smoke outputs:
- `outputs/smoke/e1/e1_metrics_summary.csv`
- `outputs/smoke/e4/e4_artifacts.csv`
- `outputs/smoke/e5/e5_artifacts.csv` (optional; set `SMOKE_ENABLE_TEXT_PROXY=1`)
- `outputs/smoke/e6/e6_artifacts.csv` (optional; set `SMOKE_ENABLE_TEXT_PROXY=1`)
- `outputs/smoke/e7/e7_input_all_methods.csv`
- `outputs/smoke/e7/e7_method_summary.csv`

## 5) Common issues

- `No module named pytest`: run `python -m pip install -r requirements.txt`
- `No module named torch`: run `python -m pip install torch torchvision`
- Slow first run on laptop: set `E1_CNN_MAX_SAMPLES_PER_SPLIT` and `E23_MAX_SAMPLES` for a smoke run first.
