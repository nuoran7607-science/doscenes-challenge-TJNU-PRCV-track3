"""
train.py
========
Main training script for the doScenes Track-2 Flow Matching model.

Features
--------
  * TensorBoard logging (step-level: loss, grad_norm, lr; epoch-level: val metrics)
  * Config saved as config.json at the start of every run
  * Gradient clipping + per-step gradient norm logging
  * Early stopping and best-model saving based on Val ADE (scene-level Q97.5)
  * Automatic run directory with timestamp
  * Saves only two checkpoints: best_model.pth (best val ADE scene-Q97.5) and last_model.pth

Usage
-----
  python train.py                                      # default config
  python train.py --epochs 100 --batch 32 --lr 5e-5   # custom 
  python train.py --resume runs/run_XXXX/checkpoints/epoch_050.pth

  tensorboard --logdir=dir --port=1234

Output layout
-------------
  runs/
  └── run_<timestamp>/
      ├── config.json           ← all hyper-parameters
      ├── tensorboard/          ← TensorBoard event files
      └── checkpoints/
          ├── best_model.pth    ← lowest val FDE so far (overwritten on improvement)
          └── last_model.pth    ← most recent epoch    (overwritten every epoch)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import make_dataloaders
from model.flow_matching import FlowMatchingConfig, FlowMatchingTrainer
from model.scorer import TrajectoryScorer, scorer_loss as scorer_loss_fn


# ══════════════════════════════════════════════════════════════════════════════
# CLI arguments
# ══════════════════════════════════════════════════════════════════════════════
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="doScenes Track-2 — Flow Matching trainer")

    # ── Paths ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--train_pkl",
        default="datasets/pre_processed_data/train_track2.pkl",
        help="Default matches dataset_process_official.py OUTPUT_DIR",
    )
    p.add_argument(
        "--val_pkl",
        default="datasets/pre_processed_data/val_track2.pkl",
        help="Default matches dataset_process_official.py OUTPUT_DIR",
    )
    p.add_argument("--log_dir",    default="runs",
                   help="Root directory for all run artefacts")
    p.add_argument("--run_name",   default='test',
                   help="Custom run name (default: auto-generated timestamp)")
    p.add_argument("--resume",     default=None,
                   help="Path to a .pth checkpoint to resume training from")

    # ── Training loop ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",          type=int,   default=120)
    p.add_argument("--batch",           type=int,   default=64)
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--val_every",       type=int,   default=1,
                   help="Run full validation every N epochs")
    p.add_argument("--val_n_steps",     type=int,   default=30,
                   help="Euler steps used during validation sampling")

    # ── Optimiser ──────────────────────────────────────────────────────────────
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--grad_clip",       type=float, default=1.0,
                   help="Max gradient L2-norm (0 = disabled)")
    p.add_argument("--warmup_epochs",   type=int,   default=5)

    # ── Model (mirrors FlowMatchingConfig fields) ──────────────────────────────
    p.add_argument("--hidden_dim",        type=int,   default=128)
    p.add_argument("--embed_dim",         type=int,   default=256)
    p.add_argument("--n_hist_layers",     type=int,   default=2)
    p.add_argument("--n_unfreeze_bert",   type=int,   default=2)
    p.add_argument("--n_layers_velocity", type=int,   default=3)
    p.add_argument("--dropout",           type=float, default=0.1)

    # ── Monitoring ────────────────────────────────────────────────────────────
    p.add_argument("--log_every_n_steps", type=int,   default=50,
                   help="Log step-level metrics (loss, grad_norm, lr) every N steps")

    # ── Early stopping ────────────────────────────────────────────────────────
    p.add_argument("--early_stop_patience",   type=int,   default=100,
                   help="Stop if val FDE does not improve for this many epochs")
    p.add_argument("--early_stop_min_delta",  type=float, default=1e-4,
                   help="Minimum FDE improvement to be counted as improvement")

    # ── Misc ───────────────────────────────────────────────────────────────────
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--filter_no_instruction", action="store_true")

    # ── EMA (Exponential Moving Average) ─────────────────────────────────────
    p.add_argument("--ema_decay",
                   type=float, default=0.9999,
                   help="EMA decay rate. Higher = smoother "
                        "average. Typical range: 0.999 – 0.9999.")

    # ── Classifier-Free Guidance (CFG) ────────────────────────────────────────
    # Ablation:
    #   CFG disabled (baseline): --cfg_dropout_prob 0.0 --cfg_guidance_scale 1.0
    #   CFG enabled  (proposed):  --cfg_dropout_prob 0.15 --cfg_guidance_scale 1.5
    p.add_argument("--cfg_dropout_prob",   type=float, default=0.15,
                   help="Instruction null-dropout prob during training. "
                        "0.0 = CFG disabled (ablation baseline).")
    p.add_argument("--cfg_guidance_scale", type=float, default=1.5,
                   help="CFG guidance weight w at inference. "
                        "1.0 = standard sampling (CFG off).")

    # ── Trajectory Scorer ──────────────────────────────────────────────────────
    p.add_argument("--scorer_k",           type=int,   default=20,
                   help="Number of correlated candidates generated per scene.")
    p.add_argument("--scorer_noise_scale", type=float, default=0.5,
                   help="Anchored-noise diversity α. 0=identical, 1=independent.")
    p.add_argument("--scorer_temperature", type=float, default=0.5,
                   help="Soft-target temperature τ for scorer_loss.")
    p.add_argument("--scorer_n_steps",     type=int,   default=20,
                   help="Euler steps for correlated sampling during scorer training.")
    p.add_argument("--scorer_start_epoch", type=int,   default=40,
                   help="Epoch from which Scorer starts training (two-stage). "
                        "Before this epoch only the Flow model is updated, giving "
                        "it time to converge before the Scorer sees candidates. "
                        "Default 40 matches observed Flow convergence point. "
                        "Set to 1 to restore the original concurrent behaviour.")
    # ── Device ────────────────────────────────────────────────────────────────
    p.add_argument("--cuda", type=int, default=0,
                   help="GPU index to use (e.g. 0, 1, 2 …). "
                        "Falls back to CPU if CUDA is unavailable. Default: 0.")

    # ── Auto-submission (训练结束后自动跑 test 并写 submission.csv) ──────────
    p.add_argument(
        "--test_pkl",
        default="datasets/pre_processed_data/test_track2.pkl",
        help="测试集 pkl，由 dataset_process_official.py 生成。默认匹配预处理"
             "脚本输出位置。",
    )
    p.add_argument(
        "--auto_submit", action=argparse.BooleanOptionalAction, default=True,
        help="训练结束后是否自动加载 best_model.pth 跑 test 集并写出 "
             "submission.csv (默认 True，可用 --no-auto_submit 关闭)",
    )
    p.add_argument(
        "--submit_baseline", action=argparse.BooleanOptionalAction, default=True,
        help="是否同时输出 history-only baseline submission "
             "(instruction='') (默认 True)",
    )
    p.add_argument(
        "--submit_n_steps", type=int, default=50,
        help="提交推理的 Euler ODE 积分步数 (default: 50, 比训练验证更精)",
    )
    p.add_argument(
        "--submit_use_scorer", action="store_true",
        help="提交时使用 K 候选 + Scorer 选优 (默认关：单次预测最稳)",
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average) of model weights
# ══════════════════════════════════════════════════════════════════════════════
class EMAModel:
    """
    Maintains an exponential moving average (EMA) of a model's trainable
    parameters.  EMA weights generalize better than the raw SGD/Adam weights
    because they smooth out the noise introduced by mini-batch updates.

    Algorithm
    ---------
    After each optimizer step:
        shadow[k]  ←  decay × shadow[k]  +  (1 − decay) × param[k]

    At validation time, the live weights are temporarily replaced by the EMA
    shadow weights; after validation the live weights are restored so that
    training can continue normally.

    Parameters
    ----------
    model : nn.Module
        The model whose trainable parameters to track.
    decay : float
        EMA decay rate.  Typical values: 0.999 – 0.9999.
        Higher = slower update = smoother average.
        At 0.9999 the EMA weights represent roughly the last
        1/(1−0.9999) ≈ 10 000 optimizer steps.

    EMA is always enabled in the cleaned training pipeline; only its decay is
    configurable.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay   = decay
        # Shadow copy: {param_name: shadow_tensor}
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        # Temporary backup used during apply_shadow / restore cycle
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow weights with the latest model weights.
        Must be called once after every ``optimizer.step()``.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Replace the model's live weights with EMA shadow weights.
        Always pair with ``restore()`` afterwards.
        """
        self._backup.clear()
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore the original (non-EMA) weights after validation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Early stopping
# ══════════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    Stops training when a monitored metric (Val FDE) stops improving.

    Parameters
    ----------
    patience  : epochs to wait without improvement before stopping
    min_delta : minimum absolute improvement to reset the patience counter
    mode      : "min" — lower is better (FDE, ADE); "max" — higher is better
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self._best      = math.inf if mode == "min" else -math.inf
        self._counter   = 0
        self._improved  = False

    @property
    def improved(self) -> bool:
        """True if the last call to step() recorded an improvement."""
        return self._improved

    @property
    def best(self) -> float:
        return self._best

    def step(self, metric: float) -> bool:
        """
        Call once per validation epoch.

        Returns
        -------
        True if training should stop, False otherwise.
        """
        if self.mode == "min":
            self._improved = metric < self._best - self.min_delta
        else:
            self._improved = metric > self._best + self.min_delta

        if self._improved:
            self._best    = metric
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
# Plaintext metrics logger  (metrics.jsonl + summary.json)
# ══════════════════════════════════════════════════════════════════════════════
class MetricsLogger:
    """
    Writes one JSON object per epoch to <run_dir>/metrics.jsonl and a final
    summary to <run_dir>/summary.json.

    Format of metrics.jsonl
    -----------------------
    Each line is a self-contained JSON record:
        {"epoch": 1, "train_loss": 0.1234, "lr": 1e-4,
         "val_loss": 0.1100, "val_ade_q975": 0.5678, "val_fde_q975": 0.9012,
         "val_ade_raw": 0.5400, "val_fde_raw": 0.8800,
         "val_ade_scene_mean": 0.5500, "val_fde_scene_mean": 0.8900,
         "scorer_loss": 0.0312, "stage": 1,
         "n_scenes": 200, "n_windows": 1400, "elapsed_s": 42.3}

    This file can be read directly with the Read tool and parsed with json.loads().
    """

    def __init__(self, run_dir: Path) -> None:
        self._path   = run_dir / "metrics.jsonl"
        self._summary_path = run_dir / "summary.json"
        self._fh     = open(self._path, "w", encoding="utf-8")
        print(f"[metrics] plaintext log → {self._path}")

    def log(self, record: dict) -> None:
        """Append one record (dict) as a JSON line."""
        self._fh.write(json.dumps(record, default=float) + "\n")
        self._fh.flush()

    def write_summary(self, summary: dict) -> None:
        """Write a final summary dict to summary.json (human-readable)."""
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"[metrics] summary      → {self._summary_path}")

    def close(self) -> None:
        self._fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Run directory + config saving
# ══════════════════════════════════════════════════════════════════════════════
def create_run_dir(log_dir: str, run_name: Optional[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name      = run_name or f"run_{timestamp}"
    run_dir   = Path(log_dir) / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, args: argparse.Namespace, cfg: FlowMatchingConfig) -> None:
    """Serialize all hyper-parameters to <run_dir>/config.json."""
    config_dict = {
        "training": vars(args),
        "model":    asdict(cfg),
    }
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[config] saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════
def compute_ade_fde(
    pred: torch.Tensor,   # [B, T, 2]
    gt:   torch.Tensor,   # [B, T, 2]
) -> Dict[str, float]:
    """
    ADE : mean L2 across all timesteps and samples.
    FDE : mean L2 at the final timestep only.
    """
    l2  = torch.norm(pred - gt, dim=-1)   # [B, T]
    ade = l2.mean().item()
    fde = l2[:, -1].mean().item()
    return {"ade": ade, "fde": fde}


def compute_per_sample_ade_fde(
    pred: torch.Tensor,   # [B, T, 2]
    gt:   torch.Tensor,   # [B, T, 2]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-sample ADE and FDE tensors (not averaged).

    Returns
    -------
    ade_per_sample : FloatTensor [B]  — mean L2 over T for each sample
    fde_per_sample : FloatTensor [B]  — L2 at final timestep for each sample
    """
    l2 = torch.norm(pred - gt, dim=-1)   # [B, T]
    return l2.mean(dim=-1), l2[:, -1]    # [B], [B]


# ══════════════════════════════════════════════════════════════════════════════
# Official local-frame metrics (mirror of offical_github/metrics.py, GPU-friendly)
# ══════════════════════════════════════════════════════════════════════════════
def _headings_with_origin(traj: torch.Tensor) -> torch.Tensor:
    """
    Per-step heading derived from waypoint displacements with a [0, 0] origin
    prepended so step 0 measures the heading from local anchor [0, 0] to the
    first predicted waypoint.

    Mirrors ``offical_github/metrics.py::_headings_from_waypoints``.

    Parameters
    ----------
    traj : FloatTensor [..., T, 2]   trajectory in local frame (anchor = origin).

    Returns
    -------
    headings : FloatTensor [..., T]  per-step headings in radians.
    """
    origin = torch.zeros(*traj.shape[:-2], 1, 2, device=traj.device, dtype=traj.dtype)
    pts = torch.cat([origin, traj], dim=-2)        # [..., T+1, 2]
    deltas = pts[..., 1:, :] - pts[..., :-1, :]    # [..., T, 2]
    return torch.atan2(deltas[..., 1], deltas[..., 0])


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap signed angle differences to (-pi, pi]."""
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def compute_per_sample_metrics_local(
    pred: torch.Tensor,    # [B, T, 2]   local frame, anchor = origin
    gt:   torch.Tensor,    # [B, T, 2]
) -> Dict[str, torch.Tensor]:
    """
    Per-sample reproduction of every **local-frame** metric reported by
    ``offical_github/metrics.py::compute_ego_metrics``:

      ade_2s / ade_4s / ade_6s : mean L2 over the first 4 / 8 / 12 steps.
      fde                       : L2 at the final step.
      miss_rate                 : 1 if any step exceeds 2 m, else 0.
      speed_error               : mean abs per-step speed error (m / 0.5 s).
      ahe                       : mean abs per-step heading error (rad).
      fhe                       : abs heading error at the final step (rad).

    Map-aware metrics (offroad / offroad_rate / offyaw) require ``NuScenesMap``
    and are therefore not computed here — use ``compute_ego_metrics`` from
    ``offical_github.metrics`` in the dedicated evaluation script instead.

    All returned tensors have shape ``[B]`` and live on the same device as
    ``pred`` so they can be aggregated downstream without extra data movement.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred {tuple(pred.shape)} vs gt {tuple(gt.shape)} mismatch")

    B, T, _ = pred.shape
    l2 = torch.norm(pred - gt, dim=-1)             # [B, T]

    steps_2s = min(4, T)
    steps_4s = min(8, T)
    steps_6s = T

    ade_2s = l2[:, :steps_2s].mean(dim=-1)
    ade_4s = l2[:, :steps_4s].mean(dim=-1)
    ade_6s = l2[:, :steps_6s].mean(dim=-1)
    fde    = l2[:, -1]
    miss_rate = (l2.amax(dim=-1) >= 2.0).float()

    # Per-step speed (with origin prepended): d(pred[t], pred[t-1]) etc.
    origin = torch.zeros(B, 1, 2, device=pred.device, dtype=pred.dtype)
    pred_pts = torch.cat([origin, pred], dim=1)
    gt_pts   = torch.cat([origin, gt],   dim=1)
    pred_speeds = torch.norm(pred_pts[:, 1:] - pred_pts[:, :-1], dim=-1)   # [B, T]
    gt_speeds   = torch.norm(gt_pts[:, 1:]   - gt_pts[:, :-1],   dim=-1)   # [B, T]
    speed_error = (pred_speeds - gt_speeds).abs().mean(dim=-1)             # [B]

    # Heading metrics (rad), wrap diff to (-pi, pi].
    pred_h = _headings_with_origin(pred)            # [B, T]
    gt_h   = _headings_with_origin(gt)              # [B, T]
    diff   = _wrap_to_pi(pred_h - gt_h)             # [B, T]
    ahe    = diff.abs().mean(dim=-1)                # [B]
    fhe    = diff[:, -1].abs()                      # [B]

    return {
        "ade_2s":      ade_2s,
        "ade_4s":      ade_4s,
        "ade_6s":      ade_6s,
        "fde":         fde,
        "miss_rate":   miss_rate,
        "speed_error": speed_error,
        "ahe":         ahe,
        "fhe":         fhe,
    }


def q975_filtered_mean(values: torch.Tensor) -> float:
    """
    Q97.5-filtered mean: keep values <= the 97.5th percentile, then average.

    Intended for **one scalar per scene** (mean ADE / mean FDE within that scene).
    This matches the doScenes baseline spec: remove the top ~2.5% highest-error
    *scenes*, then report the mean over the rest.
    """
    if values.numel() == 0:
        return float("nan")
    threshold = torch.quantile(values.float(), 0.975)
    kept = values[values <= threshold]
    return kept.mean().item() if kept.numel() > 0 else values.mean().item()


def aggregate_per_scene_means(
    scene_names: List[str],
    ade_per_sample: torch.Tensor,
    fde_per_sample: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Average per-window ADE/FDE within each nuScenes scene → one value per scene.

    Returns
    -------
    ade_scene : FloatTensor [N_scenes]
    fde_scene : FloatTensor [N_scenes]
    """
    ade_lists: Dict[str, List[float]] = defaultdict(list)
    fde_lists: Dict[str, List[float]] = defaultdict(list)
    for i, name in enumerate(scene_names):
        ade_lists[name].append(float(ade_per_sample[i].item()))
        fde_lists[name].append(float(fde_per_sample[i].item()))
    keys = sorted(ade_lists.keys())
    ade_scene = torch.tensor(
        [sum(ade_lists[k]) / len(ade_lists[k]) for k in keys],
        dtype=torch.float32,
    )
    fde_scene = torch.tensor(
        [sum(fde_lists[k]) / len(fde_lists[k]) for k in keys],
        dtype=torch.float32,
    )
    return ade_scene, fde_scene


def aggregate_metrics_per_scene(
    scene_names: List[str],
    metrics_per_sample: Dict[str, torch.Tensor],
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """
    Group per-window metrics by ``scene_name`` and average within each scene.

    Returns
    -------
    keys           : List[str]                — unique scene names in sorted order.
    scene_metrics  : Dict[str, FloatTensor[N_scenes]]
                     One scene-level mean per metric, aligned with ``keys``.
    """
    if not scene_names:
        return [], {k: torch.empty(0) for k in metrics_per_sample}

    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {k: [] for k in metrics_per_sample}
    )
    for i, name in enumerate(scene_names):
        for k, v in metrics_per_sample.items():
            buckets[name][k].append(float(v[i].item()))

    keys = sorted(buckets.keys())
    scene_metrics: Dict[str, torch.Tensor] = {}
    for k in metrics_per_sample:
        scene_metrics[k] = torch.tensor(
            [sum(buckets[name][k]) / len(buckets[name][k]) for name in keys],
            dtype=torch.float32,
        )
    return keys, scene_metrics


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all gradients that require grad."""
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_sq += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total_sq)


# ══════════════════════════════════════════════════════════════════════════════
# Training epoch
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model:               FlowMatchingTrainer,
    loader,
    optimiser:           torch.optim.Optimizer,
    device:              torch.device,
    grad_clip:           float,
    epoch:               int,
    writer:              SummaryWriter,
    global_step:         int,
    log_every_n_steps:   int,
    scorer:              Optional[TrajectoryScorer]      = None,
    scorer_optimiser:    Optional[torch.optim.Optimizer] = None,
    scorer_k:            int   = 20,
    scorer_n_steps:      int   = 20,
    scorer_noise_scale:  float = 0.5,
    scorer_temperature:  float = 0.5,
    scorer_start_epoch:  int   = 40,
    ema:                 Optional["EMAModel"]            = None,
) -> Tuple[Dict[str, float], int]:
    """
    Returns
    -------
    metrics     : {"loss": float} or {"loss": float, "scorer_loss": float}
    global_step : updated step counter

    Two-stage training
    ------------------
    Stage 1 (epoch < scorer_start_epoch):
        Only Step A (Flow Matching update) is executed.  The Scorer is skipped
        so that it only sees high-quality candidates once the Flow model has
        converged to a stable solution.
    Stage 2 (epoch >= scorer_start_epoch):
        Both Step A (Flow update) and Step B (Scorer update) run each batch,
        matching the original concurrent behaviour.

    EMA + Scorer
    ------------
    When ``ema`` is not None, Step B temporarily applies EMA shadow weights to
    the flow model before ``sample_correlated`` / ``condition_encoder``, then
    restores live weights.  This matches validation (which also evaluates with
    EMA) and avoids training the Scorer on live features but testing on EMA
    features.
    """
    # One-time console notice at the Stage 1 → Stage 2 transition
    scorer_active = (
        scorer is not None
        and scorer_optimiser is not None
        and epoch >= scorer_start_epoch
    )
    if scorer is not None and epoch == scorer_start_epoch:
        print(
            f"\n[stage]  ── Stage 2 begins at epoch {epoch}: "
            f"Scorer training now active (Flow model has converged). ──\n"
        )

    model.train()
    total_loss         = 0.0
    total_scorer_loss  = 0.0
    n_batches          = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True)

    for batch in pbar:
        history      = batch["history"].to(device)      # [B, T_hist, 2]
        future       = batch["future"].to(device)        # [B, T_fut,  2]
        instructions = batch["instruction"]              # List[str]

        # ── Step A: Flow Matching update ─────────────────────────────────────
        optimiser.zero_grad(set_to_none=True)

        out  = model(future, history, instructions)
        loss = out["loss"]

        loss.backward()

        # Gradient norm before clipping
        grad_norm = compute_grad_norm(model)

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=grad_clip,
            )

        optimiser.step()

        # ── EMA update (flow model only, after every optimizer step) ─────────
        if ema is not None:
            ema.update(model)

        # ── Step B: Scorer update (Stage 2 only, flow model frozen) ──────────
        s_loss_val = 0.0
        if scorer_active:
            # When EMA is enabled, generate candidates with the same weights as
            # validation (shadow); otherwise use live weights (unchanged baseline).
            if ema is not None:
                ema.apply_shadow(model)

            model.eval()
            with torch.no_grad():
                candidates = model.sample_correlated(
                    history, instructions,
                    n_samples=scorer_k,
                    n_steps=scorer_n_steps,
                    noise_scale=scorer_noise_scale,
                )                                        # [B, K, T_fut, 2]
                c = model.condition_encoder(history, instructions)  # [B, D]

            if ema is not None:
                ema.restore(model)

            model.train()

            # Only the scorer's parameters receive gradients here
            scorer.train()
            scorer_optimiser.zero_grad(set_to_none=True)

            scores  = scorer(c, candidates)              # [B, K]
            s_loss  = scorer_loss_fn(
                scores, candidates, future,
                temperature=scorer_temperature,
            )
            s_loss.backward()
            scorer_optimiser.step()

            s_loss_val = s_loss.item()
            total_scorer_loss += s_loss_val

        global_step += 1
        total_loss  += loss.item()
        n_batches   += 1

        postfix = {"loss": f"{loss.item():.4f}", "gnorm": f"{grad_norm:.3f}"}
        if scorer_active:
            postfix["s_loss"] = f"{s_loss_val:.4f}"
        pbar.set_postfix(**postfix)

        # ── Step-level TensorBoard logging ───────────────────────────────────
        if global_step % log_every_n_steps == 0:
            current_lr = optimiser.param_groups[0]["lr"]
            writer.add_scalar("step/train_loss",  loss.item(), global_step)
            writer.add_scalar("step/grad_norm",   grad_norm,   global_step)
            writer.add_scalar("step/lr",          current_lr,  global_step)
            if scorer_active:
                writer.add_scalar("step/scorer_loss", s_loss_val, global_step)

    n = max(n_batches, 1)
    metrics: Dict[str, float] = {"loss": total_loss / n}
    if scorer_active:
        metrics["scorer_loss"] = total_scorer_loss / n
    return metrics, global_step


# ══════════════════════════════════════════════════════════════════════════════
# Validation epoch
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(
    model:               FlowMatchingTrainer,
    loader,
    device:              torch.device,
    epoch:               int,
    n_steps:             int   = 30,
    guidance_scale:      float = 1.0,
    scorer:              TrajectoryScorer = None,
    scorer_k:            int   = 20,
    scorer_n_steps:      int   = 20,
    scorer_noise_scale:  float = 0.5,
) -> Dict[str, float]:
    """
    Validation loop using correlated sampling and TrajectoryScorer selection.

    ADE/FDE reporting follows the doScenes baseline spec:
        1. Mean ADE (and FDE) within each nuScenes **scene** over all val windows.
        2. Q97.5 filter on those **scene-level** scores (drop ~top 2.5% scenes).
        3. Report the mean of the remaining scenes.

    Also returns sample-level means (all windows pooled) for reference.
    """
    model.eval()
    scorer.eval()

    total_loss = 0.0
    n_batches  = 0

    # 累积所有官方 local-frame 指标 + scene 名称
    metric_keys = ["ade_2s", "ade_4s", "ade_6s", "fde",
                   "miss_rate", "speed_error", "ahe", "fhe"]
    accum: Dict[str, List[torch.Tensor]] = {k: [] for k in metric_keys}
    flat_scene_names: List[str] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, dynamic_ncols=True)

    for batch in pbar:
        history      = batch["history"].to(device)
        future       = batch["future"].to(device)
        instructions = batch["instruction"]
        sn = batch["scene_name"]
        scene_names = [sn] if isinstance(sn, str) else list(sn)

        # CFM loss (always computed regardless of scorer / WTA)
        out         = model(future, history, instructions)
        total_loss += out["loss"].item()

        # Correlated sampling → scorer selects the best complete trajectory.
        candidates = model.sample_correlated(
            history, instructions,
            n_samples=scorer_k,
            n_steps=scorer_n_steps,
            noise_scale=scorer_noise_scale,
            guidance_scale=guidance_scale,
        )                                               # [B, K, T_fut, 2]
        c = model.condition_encoder(history, instructions)
        pred, _ = scorer.select_best(c, candidates)     # [B, T_fut, 2]

        # 官方 local-frame 指标 (per sample)
        m = compute_per_sample_metrics_local(pred, future)
        for k in metric_keys:
            accum[k].append(m[k].detach().cpu())

        flat_scene_names.extend(scene_names)
        n_batches += 1

        pbar.set_postfix(
            ade=f"{m['ade_6s'].mean().item():.3f}",
            fde=f"{m['fde'].mean().item():.3f}",
        )

    n = max(n_batches, 1)
    if n_batches == 0:
        nan = float("nan")
        empty: Dict[str, float] = {"loss": nan, "n_scenes_val": 0.0, "n_windows_val": 0.0}
        # 兼容旧字段
        empty.update({"ade": nan, "fde": nan, "ade_raw": nan, "fde_raw": nan,
                      "ade_scene_mean_raw": nan, "fde_scene_mean_raw": nan})
        for k in metric_keys:
            empty[f"{k}_raw"]        = nan
            empty[f"{k}_scene_mean"] = nan
            empty[f"{k}_q975"]       = nan
        return empty

    # ── Sample-level (all windows) ────────────────────────────────────────────
    flat: Dict[str, torch.Tensor] = {k: torch.cat(v) for k, v in accum.items()}  # [N_windows]
    n_windows = int(next(iter(flat.values())).numel())

    # ── Scene-level: mean per scene, then Q97.5 over scenes (baseline) ──────
    _, scene_metrics = aggregate_metrics_per_scene(flat_scene_names, flat)
    n_scenes = int(next(iter(scene_metrics.values())).numel()) if scene_metrics else 0

    result: Dict[str, float] = {"loss": total_loss / n}
    for k in metric_keys:
        result[f"{k}_raw"]        = float(flat[k].mean().item())
        result[f"{k}_scene_mean"] = float(scene_metrics[k].mean().item())
        result[f"{k}_q975"]       = q975_filtered_mean(scene_metrics[k])

    # ── 兼容字段 (best-model selection 仍然使用 ADE q97.5) ───────────────────
    # 这里的 ade ≡ ade_6s_q975 (scene 级 q97.5 ADE)，与官方 leaderboard 主指标
    # ade_6s 一致；fde ≡ fde_q975 (scene 级 q97.5 FDE)。
    result["ade"]                = result["ade_6s_q975"]
    result["fde"]                = result["fde_q975"]
    result["ade_raw"]            = result["ade_6s_raw"]
    result["ade_scene_mean_raw"] = result["ade_6s_scene_mean"]
    result["fde_scene_mean_raw"] = result["fde_scene_mean"]
    result["n_scenes_val"]       = float(n_scenes)
    result["n_windows_val"]      = float(n_windows)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(
    ckpt_dir:         Path,
    filename:         str,
    epoch:            int,
    model:            FlowMatchingTrainer,
    optimiser:        torch.optim.Optimizer,
    scheduler,
    metrics:          Dict,
    scorer:           Optional[TrajectoryScorer]      = None,
    scorer_optimiser: Optional[torch.optim.Optimizer] = None,
    ema:              Optional["EMAModel"]             = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics":         metrics,
        "cfg":             model.cfg,
    }
    if scorer is not None:
        payload["scorer_state"] = scorer.state_dict()
    if scorer_optimiser is not None:
        payload["scorer_optimiser_state"] = scorer_optimiser.state_dict()
    if ema is not None:
        payload["ema_state"] = ema.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path:      str | Path,
    model:     FlowMatchingTrainer,
    optimiser: torch.optim.Optimizer,
    scheduler,
    device:    torch.device,
) -> int:
    ckpt        = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[resume] Loaded '{path}'  →  starting from epoch {start_epoch}")
    return start_epoch


# ══════════════════════════════════════════════════════════════════════════════
# LR warmup helper
# ══════════════════════════════════════════════════════════════════════════════
def warmup_lr_lambda(warmup_epochs: int):
    def _fn(epoch: int) -> float:
        return float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    return _fn


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = get_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir  = create_run_dir(args.log_dir, args.run_name)
    ckpt_dir = run_dir / "checkpoints"
    tb_dir   = run_dir / "tensorboard"
    print(f"[run]    {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[data]   loading datasets …")
    train_loader, val_loader = make_dataloaders(
        train_pkl=args.train_pkl,
        val_pkl=args.val_pkl,
        batch_size=args.batch,
        num_workers=args.num_workers,
        filter_no_instruction=args.filter_no_instruction,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[data]   train batches: {len(train_loader)}  |  val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = FlowMatchingConfig(
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_hist_layers=args.n_hist_layers,
        n_unfreeze_bert=args.n_unfreeze_bert,
        n_layers_velocity=args.n_layers_velocity,
        dropout_encoder=args.dropout,
        dropout_velocity=args.dropout,
        cfg_dropout_prob=args.cfg_dropout_prob,
        cfg_guidance_scale=args.cfg_guidance_scale,
    )
    model = FlowMatchingTrainer(cfg).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model]  total: {total:,}  |  trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    cfg_status = (
        f"ENABLED  (dropout={args.cfg_dropout_prob}, w={args.cfg_guidance_scale})"
        if args.cfg_dropout_prob > 0.0
        else "DISABLED (ablation baseline)"
    )
    print(f"[CFG]    {cfg_status}")
    print(f"[hist]   BiGRU        (bidirectional, layers={args.n_hist_layers})")
    print("[t_samp] Uniform       (t_eps from cfg)")
    print("[dual]   ENABLED")

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMAModel(model, decay=args.ema_decay)
    print(f"[EMA]    ENABLED  (decay={args.ema_decay})")

    # ── Trajectory Scorer ─────────────────────────────────────────────────────
    scorer = TrajectoryScorer(
        hidden_dim=args.hidden_dim,
        traj_dim=cfg.traj_dim,
        fut_len=cfg.fut_len,
    ).to(device)
    scorer_optimiser = AdamW(
        scorer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    s_total = sum(p.numel() for p in scorer.parameters())
    print(f"[scorer] ENABLED  params={s_total:,}  "
          f"K={args.scorer_k}  α={args.scorer_noise_scale}  "
          f"τ={args.scorer_temperature}")

    # ── Save config ───────────────────────────────────────────────────────────
    save_config(run_dir, args, cfg)

    # ── TensorBoard writer ────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(tb_dir))
    # Log hyper-parameters as text for easy reference inside TensorBoard
    writer.add_text(
        "config",
        f"```json\n{json.dumps({'training': vars(args), 'model': asdict(cfg)}, indent=2)}\n```",
    )
    print(f"[tb]     tensorboard --logdir {tb_dir}")

    # ── Plaintext metrics logger (readable without TensorBoard) ───────────────
    metrics_logger = MetricsLogger(run_dir)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=warmup_lr_lambda(args.warmup_epochs)
    )
    cosine_sched = CosineAnnealingLR(
        optimiser,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.lr * 1e-2,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs],
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimiser, scheduler, device)
        ckpt = torch.load(args.resume, map_location=device)
        # Restore scorer state if it was saved alongside the flow model
        if scorer is not None:
            if "scorer_state" in ckpt and "scorer_optimiser_state" in ckpt:
                scorer.load_state_dict(ckpt["scorer_state"])
                scorer_optimiser.load_state_dict(ckpt["scorer_optimiser_state"])
                print("[resume] Scorer state restored from checkpoint.")
        # Restore EMA state if present
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            print("[resume] EMA state restored from checkpoint.")

    # ── Early stopping & best-model tracker (criterion: Val ADE scene-Q97.5) ──
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="min",
    )
    best_ade   = math.inf
    best_epoch = -1
    global_step = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[train]  Starting: epochs {start_epoch} → {args.epochs}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────────
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimiser=optimiser,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            log_every_n_steps=args.log_every_n_steps,
            scorer=scorer,
            scorer_optimiser=scorer_optimiser,
            scorer_k=args.scorer_k,
            scorer_n_steps=args.scorer_n_steps,
            scorer_noise_scale=args.scorer_noise_scale,
            scorer_temperature=args.scorer_temperature,
            scorer_start_epoch=args.scorer_start_epoch,
            ema=ema,
        )

        # ── LR step ───────────────────────────────────────────────────────────
        scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]

        # ── Epoch-level TensorBoard: train ────────────────────────────────────
        writer.add_scalar("epoch/train_loss", train_metrics["loss"], epoch)
        writer.add_scalar("epoch/lr",         current_lr,           epoch)
        if scorer is not None:
            writer.add_scalar(
                "epoch/stage",
                2.0 if epoch >= args.scorer_start_epoch else 1.0,
                epoch,
            )

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics: Dict = {}
        should_stop = False

        if epoch % args.val_every == 0:
            # Apply EMA weights for validation (live weights stay untouched)
            if ema is not None:
                ema.apply_shadow(model)

            val_metrics = validate(
                model, val_loader, device, epoch,
                n_steps=args.val_n_steps,
                guidance_scale=args.cfg_guidance_scale,
                scorer=scorer,
                scorer_k=args.scorer_k,
                scorer_n_steps=args.scorer_n_steps,
                scorer_noise_scale=args.scorer_noise_scale,
            )

            # Restore live weights after validation
            if ema is not None:
                ema.restore(model)

            # Epoch-level TensorBoard: val
            writer.add_scalar("epoch/val_loss",             val_metrics["loss"],               epoch)
            writer.add_scalar("epoch/val_ade_scene_q975",   val_metrics["ade"],                epoch)
            writer.add_scalar("epoch/val_fde_scene_q975",   val_metrics["fde"],                epoch)
            writer.add_scalar("epoch/val_ade_sample_mean",  val_metrics["ade_raw"],            epoch)
            writer.add_scalar("epoch/val_ade_scene_mean",   val_metrics["ade_scene_mean_raw"], epoch)
            writer.add_scalar("epoch/val_fde_scene_mean",   val_metrics["fde_scene_mean_raw"], epoch)
            writer.add_scalar("epoch/val_n_scenes",         val_metrics["n_scenes_val"],       epoch)
            writer.add_scalar("epoch/val_n_windows",        val_metrics["n_windows_val"],      epoch)

            # 官方完整 local-frame 指标 (ade_2s/4s/6s, fde, miss_rate,
            # speed_error, ahe, fhe) 各自的 sample-mean / scene-mean / scene-q975
            for _key in ("ade_2s", "ade_4s", "ade_6s", "fde",
                         "miss_rate", "speed_error", "ahe", "fhe"):
                writer.add_scalar(f"val_official/{_key}_sample_mean",
                                  val_metrics[f"{_key}_raw"], epoch)
                writer.add_scalar(f"val_official/{_key}_scene_mean",
                                  val_metrics[f"{_key}_scene_mean"], epoch)
                writer.add_scalar(f"val_official/{_key}_scene_q975",
                                  val_metrics[f"{_key}_q975"], epoch)

            if "scorer_loss" in train_metrics:
                writer.add_scalar("epoch/scorer_loss", train_metrics["scorer_loss"], epoch)

            # Overlay train vs val loss on the same chart
            writer.add_scalars(
                "epoch/loss_comparison",
                {"train": train_metrics["loss"], "val": val_metrics["loss"]},
                epoch,
            )

            # ── Early stopping (monitors Val ADE scene-Q97.5) ───────────────────
            should_stop = early_stopper.step(val_metrics["ade"])

            # ── Best model saving (same metric) ─────────────────────────────────
            if early_stopper.improved:
                best_ade   = val_metrics["ade"]
                best_epoch = epoch
                path = save_checkpoint(
                    ckpt_dir, "best_model.pth",
                    epoch, model, optimiser, scheduler,
                    {**train_metrics, **val_metrics},
                    scorer=scorer,
                    scorer_optimiser=scorer_optimiser,
                    ema=ema,
                )
                print(f"  ★ New best ADE(scene,q97.5)={best_ade:.4f}  →  {path}")

        # ── Console log ───────────────────────────────────────────────────────
        elapsed = time.time() - t0
        val_str = (
            f"  val_loss={val_metrics['loss']:.4f}"
            f"  ADE6s(q97.5)={val_metrics['ade']:.4f}"
            f"  FDE(q97.5)={val_metrics['fde']:.4f}"
            f"  ADE2s(raw)={val_metrics['ade_2s_raw']:.4f}"
            f"  ADE4s(raw)={val_metrics['ade_4s_raw']:.4f}"
            f"  miss={val_metrics['miss_rate_raw']:.3f}"
            f"  ahe={val_metrics['ahe_raw']:.3f}"
            f"  scenes={int(val_metrics['n_scenes_val'])}"
        ) if val_metrics else ""

        stage_id  = 2 if (scorer is not None and epoch >= args.scorer_start_epoch) else 1
        stage_str = f"  [Stage{stage_id}]" if scorer is not None else ""

        print(
            f"Epoch {epoch:03d}/{args.epochs}"
            f"  train_loss={train_metrics['loss']:.4f}"
            f"{val_str}"
            f"  lr={current_lr:.2e}"
            f"  [{elapsed:.1f}s]"
            f"{stage_str}"
        )

        # ── Plaintext metrics log (one JSON line per epoch) ───────────────────
        log_record: Dict = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "lr":         current_lr,
            "stage":      stage_id,
            "elapsed_s":  round(elapsed, 1),
        }
        if "scorer_loss" in train_metrics:
            log_record["scorer_loss"] = train_metrics["scorer_loss"]
        if val_metrics:
            log_record.update({
                "val_loss":            val_metrics["loss"],
                "val_ade_q975":        val_metrics["ade"],
                "val_fde_q975":        val_metrics["fde"],
                "val_ade_raw":         val_metrics["ade_raw"],
                "val_fde_raw":         val_metrics["fde_raw"],
                "val_ade_scene_mean":  val_metrics["ade_scene_mean_raw"],
                "val_fde_scene_mean":  val_metrics["fde_scene_mean_raw"],
                "n_scenes":            int(val_metrics["n_scenes_val"]),
                "n_windows":           int(val_metrics["n_windows_val"]),
                "is_best":             bool(early_stopper.improved),
            })
            # 官方完整 local-frame 指标 (sample-mean / scene-mean / scene-q975)
            for _key in ("ade_2s", "ade_4s", "ade_6s", "fde",
                         "miss_rate", "speed_error", "ahe", "fhe"):
                log_record[f"val_{_key}_raw"]        = val_metrics[f"{_key}_raw"]
                log_record[f"val_{_key}_scene_mean"] = val_metrics[f"{_key}_scene_mean"]
                log_record[f"val_{_key}_q975"]       = val_metrics[f"{_key}_q975"]
        metrics_logger.log(log_record)

        # ── Last-epoch checkpoint (overwrite every epoch) ────────────────────
        save_checkpoint(
            ckpt_dir, "last_model.pth",
            epoch, model, optimiser, scheduler,
            {**train_metrics, **val_metrics},
            scorer=scorer,
            scorer_optimiser=scorer_optimiser,
            ema=ema,
        )

        # ── Early stop ────────────────────────────────────────────────────────
        if should_stop:
            print(
                f"\n[early stop] Val ADE(scene,q97.5) has not improved for {args.early_stop_patience} "
                f"consecutive validation epochs.  Stopping at epoch {epoch}."
            )
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    writer.close()

    # Write plaintext summary.json and close the JSONL logger
    metrics_logger.write_summary({
        "run_name":           args.run_name,
        "best_val_ade_q975":  best_ade,
        "best_epoch":         best_epoch,
        "total_epochs_run":   epoch,
        "run_dir":            str(run_dir.resolve()),
        "config": {
            "cfg_dropout_prob":    args.cfg_dropout_prob,
            "cfg_guidance_scale":  args.cfg_guidance_scale,
            "use_scorer":          True,
            "scorer_start_epoch":  args.scorer_start_epoch,
            "scorer_k":            args.scorer_k,
            "scorer_noise_scale":  args.scorer_noise_scale,
            "scorer_temperature":  args.scorer_temperature,
            "use_ema":             True,
            "use_dual_guidance":   True,
            "lr":                  args.lr,
            "warmup_epochs":       args.warmup_epochs,
            "epochs":              args.epochs,
            "batch":               args.batch,
            "hidden_dim":          args.hidden_dim,
            "embed_dim":           args.embed_dim,
            "n_layers_velocity":   args.n_layers_velocity,
            "n_hist_layers":       args.n_hist_layers,
            "n_unfreeze_bert":     args.n_unfreeze_bert,
            "dropout":             args.dropout,
        },
    })
    metrics_logger.close()

    print(f"\n[done]   Training complete.")
    print(f"         Best val ADE(scene,q97.5) = {best_ade:.4f}  at epoch {best_epoch}")
    print(f"         Run dir      → {run_dir.resolve()}")
    print(f"         TensorBoard  → tensorboard --logdir {tb_dir}")
    print(f"         Metrics log  → {run_dir.resolve() / 'metrics.jsonl'}")
    print(f"         Summary      → {run_dir.resolve() / 'summary.json'}")

    # ── 训练结束后自动生成 submission.csv ─────────────────────────────────────
    if args.auto_submit:
        best_ckpt = ckpt_dir / "best_model.pth"
        test_pkl  = Path(args.test_pkl)
        if not best_ckpt.exists():
            print(f"\n[submit] 跳过：best checkpoint 不存在 → {best_ckpt}")
        elif not test_pkl.exists():
            print(f"\n[submit] 跳过：test pkl 不存在 → {test_pkl}")
            print(f"[submit] 提示：先运行 'python -m preprocessing."
                  f"dataset_process_official' 生成 test_track2.pkl")
        else:
            try:
                # 局部导入避免训练阶段就触发 submit_track2 的依赖
                from submit_track2 import run_submission
                print(f"\n[submit] 训练结束 → 用 best_model.pth 跑 test 集并"
                      f"写出 submission.csv")
                paths = run_submission(
                    ckpt=best_ckpt,
                    test_pkl=test_pkl,
                    out_dir=run_dir,
                    n_steps=args.submit_n_steps,
                    batch_size=args.batch,
                    use_scorer=args.submit_use_scorer,
                    scorer_k=args.scorer_k,
                    scorer_noise_scale=args.scorer_noise_scale,
                    guidance_scale=args.cfg_guidance_scale,
                    use_ema=True,
                    baseline=args.submit_baseline,
                    device=device,
                )
                print(f"[submit] 主提交 → {paths['submission']}")
                if paths.get("baseline") is not None:
                    print(f"[submit] baseline → {paths['baseline']}")
            except Exception as e:
                print(f"\n[submit] 自动提交失败: {e}")
                print(f"[submit] 训练已完成。可稍后手动运行：")
                print(f"  python submit_track2.py --ckpt {best_ckpt} "
                      f"--test_pkl {test_pkl} --out_dir {run_dir}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
