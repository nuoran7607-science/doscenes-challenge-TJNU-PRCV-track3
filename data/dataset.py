"""
DoScenesTrack2Dataset — PyTorch Dataset for doScenes Track-2 Challenge
=======================================================================
Loads the pre-processed .pkl files produced by dataset_process.py and
returns trajectory tensors together with the natural-language instruction.

Record schema (as saved by dataset_process_official.py):
{
    "scene_name"   : str
    "scene_token"  : str
    "sample_token" : str                # anchor 帧 sample token
    "history_traj" : np.float32[H, 2]   ego-centric (may be < OBS_LEN if near scene start)
    "future_traj"  : np.float32[12, 2]  ego-centric
    "anchor_xy"    : np.float32[2]      # world-frame anchor 坐标 (米)，可选
    "anchor_yaw"   : float              # world-frame anchor 朝向 (rad)，可选
    "instructions" : List[str] | None   # legacy dataset_process.py
    "instruction"  : str                # dataset_process_official.py (per-row string; may be "")
}

__getitem__ returns a dict:
{
    "history"         : FloatTensor[OBS_LEN, 2]   front-padded to OBS_LEN
    "future"          : FloatTensor[FUT_LEN, 2]
    "instruction"     : str                        one instruction (randomly sampled)
    "has_instruction" : BoolTensor[]               False when instructions is None
    "scene_name"      : str
    "sample_token"    : str
    "anchor_xy"       : FloatTensor[2]             world-frame anchor (米)；缺失填 NaN
    "anchor_yaw"      : FloatTensor[]              world-frame anchor 朝向 (rad)；缺失填 NaN
}

usage:
from data.dataset import make_dataloaders
train_loader, val_loader = make_dataloaders(
    "processed_data/train_track2.pkl",
    "processed_data/val_track2.pkl",
    batch_size=64,
)
for batch in train_loader:
    history     = batch["history"]       # [B, 5, 2]
    future      = batch["future"]        # [B, 12, 2]
    instructions = batch["instruction"]  # List[str], 长度 B
"""

import pickle
import random
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─── Constants (must match dataset_process.py) ────────────────────────────
OBS_LEN = 4    # history frames  (2 s @ 2 Hz)
FUT_LEN = 12   # future  frames  (6 s @ 2 Hz)

FALLBACK_INSTRUCTION = ""   # used when a scene has no language annotation

# Default z-score stats for (x, y); single source for Dataset + train metrics
# TRAJ_NORM_MEAN = np.array([11.1198, -0.141], dtype=np.float32)
# TRAJ_NORM_STD = np.array([16.0587, 2.7452], dtype=np.float32)
TRAJ_NORM_MEAN = None
TRAJ_NORM_STD = None

# ══════════════════════════════════════════════════════════════════════════════
class DoScenesTrack2Dataset(Dataset):
    """
    Parameters
    ----------
    pkl_path : str | Path
        Path to train_track2.pkl or val_track2.pkl.
    obs_len : int
        Expected history length. Records shorter than this are
        zero-padded at the *front* (oldest positions set to zero).
    fut_len : int
        Expected future length. Records are truncated / zero-padded
        to exactly this length.
    instruction_mode : "random" | "first" | "all"
        "random" — sample one instruction randomly at each __getitem__
                   call (recommended for training).
        "first"  — always use the first instruction in the list
                   (deterministic; useful for evaluation).
        "all"    — return the full list of instructions as a Python list.
                   (cannot be stacked by the default DataLoader collate;
                    use a custom collate_fn in this case.)
    augment : bool
        If True, apply random horizontal flip (mirror left/right) to
        both history and future trajectories during __getitem__.
        Only meaningful for training splits.
    transform : callable | None
        Optional extra transform applied to the raw record dict before
        the tensor conversion.  Receives a copy of the raw record and
        must return a modified copy.
    filter_no_instruction : bool
        If True, drop samples without language. For official pkl this means
        ``instruction`` is missing/blank; for legacy pkl, ``instructions`` is
        None or an empty list.
    """

    def __init__(
        self,
        pkl_path: str | Path,
        obs_len: int = OBS_LEN,
        fut_len: int = FUT_LEN,
        instruction_mode: str = "random",
        augment: bool = False,
        transform: Optional[Callable] = None,
        filter_no_instruction: bool = False,
    ):
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            records: list = pickle.load(f)

        if filter_no_instruction:
            before = len(records)

            def _record_has_instruction(r: dict) -> bool:
                if "instruction" in r:
                    return bool(str(r.get("instruction", "")).strip())
                inst = r.get("instructions")
                return inst is not None and len(inst) > 0

            records = [r for r in records if _record_has_instruction(r)]
            print(f"filter_no_instruction=True: {before} → {len(records)} records")

        self.records          = records
        self.obs_len          = obs_len
        self.fut_len          = fut_len
        self.instruction_mode = instruction_mode
        self.augment          = augment
        self.transform        = transform

        # Pre-compute statistics for optional normalisation (call compute_norm_stats() to refresh)
        self._norm_mean: Optional[np.ndarray] = TRAJ_NORM_MEAN
        self._norm_std: Optional[np.ndarray] = TRAJ_NORM_STD

    # ── Normalization helpers ──────────────────────────────────────────────
    def compute_norm_stats(self):
        """
        Compute mean and std of future trajectory displacements across
        the whole dataset and store them.  Call once before training
        if you want z-score normalisation.
        """
        all_fut = np.concatenate(
            [r["future_traj"] for r in self.records], axis=0
        )   # (N*FUT_LEN, 2)
        self._norm_mean = all_fut.mean(axis=0)
        self._norm_std  = all_fut.std(axis=0) + 1e-6
        print(f"Norm stats — mean: {self._norm_mean},  std: {self._norm_std}")

    def _normalise(self, traj: torch.Tensor) -> torch.Tensor:
        if self._norm_mean is None:
            return traj
        mean = torch.tensor(self._norm_mean, dtype=torch.float32)
        std  = torch.tensor(self._norm_std,  dtype=torch.float32)
        return (traj - mean) / std

    def denormalise(self, traj: torch.Tensor) -> torch.Tensor:
        """Inverse of z-score normalisation (call on model output)."""
        if self._norm_mean is None:
            return traj
        mean = torch.tensor(self._norm_mean, dtype=torch.float32).to(traj.device)
        std  = torch.tensor(self._norm_std,  dtype=torch.float32).to(traj.device)
        return traj * std + mean

    # ── Private helpers ────────────────────────────────────────────────────
    @staticmethod
    def _pad_history(hist: np.ndarray, obs_len: int) -> np.ndarray:
        """
        Front-pad with zeros so that the array is exactly (obs_len, 2).
        The anchor frame T is always the *last* row.
        """
        h = hist.shape[0]
        if h >= obs_len:
            return hist[-obs_len:]          # keep the most recent obs_len frames
        pad = np.zeros((obs_len - h, 2), dtype=np.float32)
        return np.concatenate([pad, hist], axis=0)

    @staticmethod
    def _pad_future(fut: np.ndarray, fut_len: int) -> np.ndarray:
        f = fut.shape[0]
        if f >= fut_len:
            return fut[:fut_len]
        pad = np.zeros((fut_len - f, 2), dtype=np.float32)
        return np.concatenate([fut, pad], axis=0)

    @staticmethod
    def _random_flip(hist: np.ndarray, fut: np.ndarray):
        """Mirror Y axis (left ↔ right) with 50 % probability."""
        if random.random() < 0.5:
            hist = hist.copy(); hist[:, 1] *= -1
            fut  = fut.copy();  fut[:,  1] *= -1
        return hist, fut

    def _pick_instruction(self, instructions: Optional[List[str]]) -> str:
        if not instructions:                  # None or empty list
            return FALLBACK_INSTRUCTION
        if self.instruction_mode == "random":
            return random.choice(instructions)
        if self.instruction_mode == "first":
            return instructions[0]
        # "all" mode: caller must handle the list directly
        return instructions                   # type: ignore

    # ── Dataset interface ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        if self.transform is not None:
            record = self.transform(dict(record))   # shallow copy then transform

        hist_np = record["history_traj"].astype(np.float32)
        fut_np  = record["future_traj"].astype(np.float32)

        # Pad / trim to fixed sizes
        hist_np = self._pad_history(hist_np, self.obs_len)
        fut_np  = self._pad_future(fut_np,   self.fut_len)

        # Optional data augmentation (training only)
        if self.augment:
            hist_np, fut_np = self._random_flip(hist_np, fut_np)

        hist_t = torch.from_numpy(hist_np)   # [OBS_LEN, 2]
        fut_t  = torch.from_numpy(fut_np)    # [FUT_LEN, 2]

        hist_t = self._normalise(hist_t)
        fut_t  = self._normalise(fut_t)

        if "instruction" in record:
            raw = record["instruction"]
            if raw is None:
                instruction = FALLBACK_INSTRUCTION
            elif isinstance(raw, str):
                instruction = raw
            else:
                instruction = str(raw)
            has_instruction = bool(instruction.strip())
        else:
            instruction = self._pick_instruction(record.get("instructions"))
            inst_list = record.get("instructions")
            has_instruction = inst_list is not None and len(inst_list) > 0

        has_instruction_t = torch.tensor(has_instruction, dtype=torch.bool)

        # ── anchor (world frame) — 用于把 ego 预测反投回世界坐标 ──────────────
        # 旧 pkl 没有这两个字段时填 NaN，下游可据此判断是否能算 map 指标。
        anchor_xy_raw = record.get("anchor_xy")
        if anchor_xy_raw is None:
            anchor_xy_t = torch.tensor([float("nan"), float("nan")], dtype=torch.float32)
        else:
            anchor_xy_t = torch.as_tensor(np.asarray(anchor_xy_raw, dtype=np.float32))
        anchor_yaw_raw = record.get("anchor_yaw")
        if anchor_yaw_raw is None:
            anchor_yaw_t = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            anchor_yaw_t = torch.tensor(float(anchor_yaw_raw), dtype=torch.float32)

        return {
            "history"         : hist_t,            # FloatTensor [OBS_LEN, 2]
            "future"          : fut_t,             # FloatTensor [FUT_LEN, 2]
            "instruction"     : instruction,       # str  (or List[str] if mode="all")
            "has_instruction" : has_instruction_t, # BoolTensor []
            "scene_name"      : record["scene_name"],
            "sample_token"    : record["sample_token"],
            "anchor_xy"       : anchor_xy_t,       # FloatTensor [2]
            "anchor_yaw"      : anchor_yaw_t,      # FloatTensor []
        }


# ══════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════════════════
def make_dataloaders(
    train_pkl: str | Path,
    val_pkl:   str | Path,
    batch_size: int  = 64,
    num_workers: int = 4,
    filter_no_instruction: bool = False,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience function — returns (train_loader, val_loader).

    The default collate_fn handles string instructions by keeping them
    as a list of strings inside the batch dict.
    """
    train_ds = DoScenesTrack2Dataset(
        train_pkl,
        instruction_mode="random",
        augment=False,
        filter_no_instruction=filter_no_instruction,
    )
    val_ds = DoScenesTrack2Dataset(
        val_pkl,
        instruction_mode="first",
        augment=False,
        filter_no_instruction=filter_no_instruction,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader