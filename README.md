# doScenes Challenge — Track 3: Ablation Study

This repository contains the solution for the [doScenes Instructed Driving Challenge](https://mi3-lab.github.io/doScenes_challenge) **Track 3 (Ablation)**.

The goal of this track is to quantify the benefit of language instructions ($\Delta$ADE). Participants must evaluate and report results for both **with-language** (instruction-conditioned) and **without-language** (history-only baseline) models under the exact same protocol.

## Project Structure

```
├── train.py                        # Training + auto-submission
├── submit_track3.py                # Standalone inference & submission
├── model/
│   ├── flow_matching.py            # OT-CFM model wrapper
│   ├── velocity_net.py             # Velocity field network
│   ├── encoders.py                 # History / language / fusion encoders
│   └── scorer.py                   # Trajectory scorer (candidate ranking)
├── data/
│   └── dataset.py                  # PyTorch dataset & dataloader
├── preprocessing/
│   ├── dataset_process_official.py # Raw nuScenes → training pkl
│   └── doscenes_dataloader.py      # nuScenes trajectory loader
└── datasets/
    ├── Annotations/                # doScenes annotation CSVs (included)
    └── pre_processed_data/         # Pre-processed pkl files (included)
        ├── train_track2.pkl
        ├── val_track2.pkl
        └── test_track2.pkl
```

## Environment

Python 3.10

```bash
pip install torch==2.6.0 transformers==4.57.3 nuscenes-devkit==1.1.11 numpy pandas tensorboard tqdm
```

## Quick Start

### Train + Auto-Submit

Training evaluates the model and automatically writes two files: submission.csv (with-language) and submission_baseline.csv (without-language) to calculate the Instruction Conditioning Gain ($\Delta$ADE).

```bash
python train.py \
    --run_name v1 \
    --epochs 120 \
    --n_unfreeze_bert 3 \
    --scorer_k 50 \
    --scorer_start_epoch 30 \
    --warmup_epochs 10 \
    --val_n_steps 20
```

Outputs are saved to `runs/v1/`:
```
runs/v1/
├── config.json
├── metrics.jsonl
├── summary.json
├── submission.csv               # Predictions WITH language instructions
├── submission_baseline.csv      # Predictions WITHOUT language (History-only)
├── tensorboard/
└── checkpoints/
    ├── best_model.pth
    └── last_model.pth
```

### Pretrained Weights

Download `best_model.pth` (635 MB) from [GitHub Releases](https://github.com/nuoran7607-science/doscenes-challenge_TJNU-PRCV/releases/tag/v1.0).

### Inference Only

Generate both the conditioned predictions and the ablation baseline from a trained checkpoint:

```bash
python submit_track3.py \
    --ckpt best_model.pth \
    --test_pkl datasets/pre_processed_data/test_track2.pkl \
    --out_dir submission \
```

Note: This script will output both submission.csv and submission_baseline.csv in the specified --out_dir.

## Data Preprocessing (Optional)

The `datasets/Annotations/` and `datasets/pre_processed_data/` folders are included in this repo. You do **not** need to run preprocessing unless you want to regenerate from scratch.

To regenerate:

1. Download [nuScenes](https://www.nuscenes.org/nuscenes) v1.0-trainval and v1.0-test **metadata** into `datasets/nuscenes_data/`:
   ```
   datasets/nuscenes_data/
   ├── v1.0-trainval_meta/
   │   └── v1.0-trainval/
   └── v1.0-test_meta/
       └── v1.0-test/
   ```

2. Run:
   ```bash
   python -m preprocessing.dataset_process_official
   ```

## Citation

If you use the doScenes dataset, please cite:

> Roy, P., Perisetla, S., Shriram, S., Krishnaswamy, H., Keskar, A., & Greer, R. (2025). doScenes: An autonomous driving dataset with natural language instruction for human interaction and vision-language navigation. In *IEEE ITSC 2025* (pp. 1651-1658). [arXiv:2412.05893](https://arxiv.org/abs/2412.05893)
