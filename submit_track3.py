"""
submit_track2.py
================
为 doScenes Instructed Driving Challenge 第 2 赛道 (Language + History) 生成
官方 leaderboard 提交所需的 ``submission.csv``。

工作流程
--------
1. 加载训练好的 best checkpoint (含 FlowMatchingTrainer, 可选 TrajectoryScorer
   与 EMA shadow 权重)。
2. 加载 ``test_track2.pkl`` (每个 v1.0-test scene 1 条 first-segment 样本)。
3. 对每条样本：

     pred = model.sample(history, instruction, n_samples=1, n_steps=N)

   *single-shot, open-loop*, 严格按官方评测协议。
4. 写出 ``submission.csv``，header 为::

     sample_token,instruction,x1,y1,x2,y2,...,x12,y12

   坐标为预测时刻 anchor 的 ego frame，单位米；x 轴指向车辆前方，y 轴指向左侧。

可选模式
--------
- ``--use_scorer``   : 跑 K 条候选 + Scorer 选优 (无 GT 依赖，scorer 在训练阶段
                       已学好如何挑最优)。
- ``--baseline``     : 同时跑 instruction="" 输出 ``submission_baseline.csv``，
                       用于 leaderboard 报告中的 "history-only baseline"。
- ``--no_ema``       : 默认会优先用 EMA shadow 权重；本开关可关闭 EMA 强制用
                       原始 (live) 权重。

CLI
---
::

    python submit_track2.py \
        --ckpt runs/v5.0/checkpoints/best_model.pth \
        --test_pkl datasets/pre_processed_data/test_track2.pkl \
        --out_dir runs/v5.0/

如需 single-call 入口，可 ``from submit_track2 import run_submission`` 然后::

    run_submission(
        ckpt="runs/v5.0/checkpoints/best_model.pth",
        test_pkl="datasets/pre_processed_data/test_track2.pkl",
        out_dir="runs/v5.0/",
        n_steps=50,
        baseline=True,
    )
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DoScenesTrack2Dataset
from model.flow_matching import FlowMatchingConfig, FlowMatchingTrainer
from model.scorer import TrajectoryScorer


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint utilities
# ══════════════════════════════════════════════════════════════════════════════
def _apply_ema_if_present(model: FlowMatchingTrainer, ckpt: dict) -> bool:
    """
    若 ckpt 中保存了 EMA shadow 权重，将其覆盖到 model 的 live 权重上。
    返回是否成功应用。

    与 train.py 中 ``EMAModel.apply_shadow`` 行为等价：覆盖所有
    requires_grad 的参数；缺失的参数保持 live 权重不动。
    """
    ema_state = ckpt.get("ema_state")
    if not ema_state:
        return False

    name_to_param = dict(model.named_parameters())
    n_applied = 0
    with torch.no_grad():
        for name, shadow in ema_state.items():
            param = name_to_param.get(name)
            if param is None or not param.requires_grad:
                continue
            param.data.copy_(shadow.to(param.device))
            n_applied += 1
    print(f"[ema] 已应用 EMA shadow 权重到 {n_applied} 个参数")
    return n_applied > 0


def load_model(
    ckpt_path: str | Path,
    device: torch.device,
    use_ema: bool = True,
    load_scorer: bool = False,
):
    """
    加载 FlowMatchingTrainer (+ 可选 TrajectoryScorer) checkpoint。

    Parameters
    ----------
    ckpt_path   : checkpoint 路径，由 train.py 的 ``save_checkpoint`` 保存。
    device      : torch device。
    use_ema     : 若 ckpt 存有 EMA shadow，则覆盖 live 权重 (默认开启，匹配
                  train.py 验证阶段的行为)。
    load_scorer : 是否加载 TrajectoryScorer 并准备给 ``--use_scorer`` 使用。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: FlowMatchingConfig = ckpt.get("cfg", FlowMatchingConfig())
    model = FlowMatchingTrainer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    if use_ema:
        _apply_ema_if_present(model, ckpt)

    model.eval()

    scorer: Optional[TrajectoryScorer] = None
    if load_scorer:
        if "scorer_state" not in ckpt:
            raise RuntimeError("checkpoint 中没有 scorer_state，无法启用 --use_scorer")
        scorer = TrajectoryScorer(
            hidden_dim=cfg.hidden_dim,
            traj_dim=cfg.traj_dim,
            fut_len=cfg.fut_len,
        ).to(device)
        scorer.load_state_dict(ckpt["scorer_state"])
        scorer.eval()

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    print(f"[ckpt] Loaded '{ckpt_path}'  (epoch {epoch})")
    if metrics:
        msg = "  ".join(
            f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)
        )
        if msg:
            print(f"[ckpt] saved metrics: {msg}")
    return model, scorer, cfg


# ══════════════════════════════════════════════════════════════════════════════
# CSV writer (strict official format)
# ══════════════════════════════════════════════════════════════════════════════
def _build_header(fut_len: int) -> List[str]:
    cols = ["sample_token", "instruction"]
    for t in range(1, fut_len + 1):
        cols.extend([f"x{t}", f"y{t}"])
    return cols


def _format_float(v: float) -> str:
    """统一 6 位小数；NaN / inf 退化为 0.0 (官方不接受非数字)。"""
    if not np.isfinite(v):
        return "0.000000"
    return f"{float(v):.6f}"


def _write_submission_rows(
    out_path: Path,
    fut_len: int,
    rows: List[dict],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = _build_header(fut_len)
    # csv.writer 自动按 RFC 4180 处理逗号/引号转义，包括 instruction 中的逗号。
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            row = [r["sample_token"], r["instruction"]]
            traj = r["pred"]   # np.ndarray [fut_len, 2]
            for t in range(fut_len):
                row.append(_format_float(traj[t, 0]))
                row.append(_format_float(traj[t, 1]))
            writer.writerow(row)
    print(f"[csv] 提交文件已写出 -> {out_path}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _infer_one_pass(
    model: FlowMatchingTrainer,
    scorer: Optional[TrajectoryScorer],
    test_pkl: str | Path,
    device: torch.device,
    batch_size: int,
    n_steps: int,
    use_scorer: bool,
    scorer_k: int,
    scorer_noise_scale: float,
    guidance_scale: Optional[float],
    instruction_override: Optional[str],
) -> List[dict]:
    """
    跑一遍 test pkl，返回每个 scene 的预测结果列表。
    每条记录：``{sample_token, instruction, pred (np.ndarray [T_fut, 2])}``。

    Parameters
    ----------
    instruction_override : 若不是 None，覆盖输入 instruction (用于跑 baseline
                           时把 instruction 强制设为空字符串)。
    """
    ds = DoScenesTrack2Dataset(test_pkl, instruction_mode="first", augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    out: List[dict] = []
    desc = "submit:cond" if instruction_override is None else "submit:baseline"
    print(f"\n[inference] {desc}  scenes={len(ds)}  n_steps={n_steps}  "
          f"use_scorer={use_scorer}  guidance_scale={guidance_scale}")

    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        history = batch["history"].to(device, non_blocking=True)         # [B, 4, 2]
        sn = batch["scene_name"]
        scene_names = [sn] if isinstance(sn, str) else list(sn)
        st = batch["sample_token"]
        sample_tokens = [st] if isinstance(st, str) else list(st)
        raw_instr = batch["instruction"]
        if isinstance(raw_instr, str):
            raw_instr = [raw_instr]
        else:
            raw_instr = list(raw_instr)

        # baseline 时强制空 instruction；条件提交时保持原 instruction
        if instruction_override is not None:
            instructions = [instruction_override] * len(raw_instr)
            instr_for_csv = [instruction_override] * len(raw_instr)
        else:
            instructions = list(raw_instr)
            instr_for_csv = list(raw_instr)

        if use_scorer and scorer is not None:
            # K 条候选 → scorer 选最优
            candidates = model.sample_correlated(
                history, instructions,
                n_samples=scorer_k,
                n_steps=n_steps,
                noise_scale=scorer_noise_scale,
                guidance_scale=guidance_scale,
            )                                                # [B, K, T_fut, 2]
            c = model.condition_encoder(history, instructions)
            pred, _ = scorer.select_best(c, candidates)      # [B, T_fut, 2]
        else:
            pred_k = model.sample(
                history, instructions,
                n_steps=n_steps,
                n_samples=1,
                guidance_scale=guidance_scale,
            )                                                # [B, 1, T_fut, 2]
            pred = pred_k[:, 0]                              # [B, T_fut, 2]

        pred_np = pred.detach().cpu().numpy()
        for i in range(history.size(0)):
            out.append({
                "scene_name":   scene_names[i],
                "sample_token": sample_tokens[i],
                "instruction":  instr_for_csv[i],
                "pred":         pred_np[i],
            })

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════
def run_submission(
    ckpt: str | Path,
    test_pkl: str | Path,
    out_dir: str | Path,
    n_steps: int = 50,
    batch_size: int = 32,
    use_scorer: bool = False,
    scorer_k: int = 50,
    scorer_noise_scale: float = 0.5,
    guidance_scale: Optional[float] = None,
    use_ema: bool = True,
    baseline: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """
    一站式入口：加载 ckpt -> 跑 test pkl -> 写出 submission.csv (+ baseline)。

    Returns
    -------
    Dict[str, Path]:
      - ``submission`` : 主提交 csv 路径
      - ``baseline``   : baseline csv 路径 (若 baseline=False 则为 None)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model, scorer, cfg = load_model(
        ckpt, device, use_ema=use_ema, load_scorer=use_scorer,
    )

    out_dir = Path(out_dir)

    # ── 主提交：使用真实 instruction ───────────────────────────────────────
    rows_cond = _infer_one_pass(
        model=model,
        scorer=scorer,
        test_pkl=test_pkl,
        device=device,
        batch_size=batch_size,
        n_steps=n_steps,
        use_scorer=use_scorer,
        scorer_k=scorer_k,
        scorer_noise_scale=scorer_noise_scale,
        guidance_scale=guidance_scale,
        instruction_override=None,
    )
    sub_path = out_dir / "submission.csv"
    _write_submission_rows(sub_path, fut_len=cfg.fut_len, rows=rows_cond)

    out: dict = {"submission": sub_path, "baseline": None}

    # ── baseline 提交：instruction = "" (history-only) ─────────────────────
    if baseline:
        rows_base = _infer_one_pass(
            model=model,
            scorer=scorer,
            test_pkl=test_pkl,
            device=device,
            batch_size=batch_size,
            n_steps=n_steps,
            use_scorer=use_scorer,
            scorer_k=scorer_k,
            scorer_noise_scale=scorer_noise_scale,
            guidance_scale=guidance_scale,
            instruction_override="",
        )
        base_path = out_dir / "submission_baseline.csv"
        _write_submission_rows(base_path, fut_len=cfg.fut_len, rows=rows_base)
        out["baseline"] = base_path

    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Track-2 submission.csv from a trained best_model"
    )
    p.add_argument("--ckpt", required=True,
                   help="best_model.pth 路径 (e.g. runs/v5.0/checkpoints/best_model.pth)")
    p.add_argument("--test_pkl",
                   default="datasets/pre_processed_data/test_track2.pkl",
                   help="test pkl 路径，由 dataset_process_official.py 生成")
    p.add_argument("--out_dir", required=True,
                   help="输出目录 (写出 submission.csv 与 submission_baseline.csv)")
    p.add_argument("--n_steps", type=int, default=50,
                   help="Euler ODE 积分步数 (default: 50)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--use_scorer", action="store_true",
                   help="开启 K-候选 + Scorer 选优；checkpoint 必须含 scorer_state")
    p.add_argument("--scorer_k", type=int, default=50)
    p.add_argument("--scorer_noise_scale", type=float, default=0.5)
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="CFG guidance scale; None = 使用 ckpt cfg 中的默认值")
    p.add_argument("--no_ema", action="store_true",
                   help="跳过 EMA 权重应用，使用 live 权重")
    p.add_argument("--no_baseline", action="store_true",
                   help="不输出 submission_baseline.csv")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)

    run_submission(
        ckpt=args.ckpt,
        test_pkl=args.test_pkl,
        out_dir=args.out_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        use_scorer=args.use_scorer,
        scorer_k=args.scorer_k,
        scorer_noise_scale=args.scorer_noise_scale,
        guidance_scale=args.guidance_scale,
        use_ema=(not args.no_ema),
        baseline=(not args.no_baseline),
    )


if __name__ == "__main__":
    main()
