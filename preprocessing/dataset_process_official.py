"""
按官方 ``offical_github/dataloader.py`` 的 (4 history + 1 anchor + 12 future) = 17
帧切分预处理 doScenes Track-2 训练 / 验证集。每个样本在 anchor 帧旋转、平移到
ego 坐标系后落盘，便于训练时直接读取，同时保留 ``anchor_xy`` / ``anchor_yaw``
（world frame）以便日后用官方 ``compute_ego_metrics`` 计算 map 相关指标或反算
世界坐标。

关键约定（与 ``offical_github/dataloader.py`` 完全对齐）
======================================================
对于一段连续 17 帧 ``ego_xy`` ::

    indices: 0  1  2  3  |  4  |  5  6  ...  16
              ─ history ─| anc | ───── future ─────
              4 帧         1 帧 |         12 帧

- ``HISTORY_LEN = 4``     —— 过去 2 s @ 2 Hz
- 1 anchor                 —— 预测时刻 ``t = 0``，独立于 history
- ``FUTURE_LEN = 12``     —— 未来 6 s @ 2 Hz

提交时 ``sample_token`` 应使用 anchor 帧的 sample token（即 ``ego_xy[i+4]`` 对应
的 nuScenes sample），与官方 ``DoScenesNuScenesDataset`` 返回的
``anchor_sample_token`` 一致。

训练 / 验证仍采用滑动窗口扩样本（每滑动 1 帧产生一条样本），可在多场景下保留
更多训练数据；leaderboard 测试集 (``v1.0-test``) 则仅用每个 scene 第一段
（``i = 0``），每个 scene 一条样本。

数据布局
========
- nuScenes trainval 元数据: ``datasets/nuscenes_data/v1.0-trainval_meta/v1.0-trainval/``
- nuScenes test 元数据    : ``datasets/nuscenes_data/v1.0-test_meta/v1.0-test/``
- doScenes 标注 CSV       : ``datasets/Annotations/``
- 输出 pkl                : ``datasets/pre_processed_data/{train,val,test}_track2.pkl``

CLI
===
::

    python -m preprocessing.dataset_process_official              # 跑所有
    python -m preprocessing.dataset_process_official --skip_test  # 只 train/val
    python -m preprocessing.dataset_process_official --skip_train_val  # 只 test
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from .doscenes_dataloader import _decode_instruction_type, load_scene_trajectory

# ─── 路径根 (Code_final/) ────────────────────────────────────────────────
# Path(__file__).resolve().parent      → preprocessing/
# Path(__file__).resolve().parent.parent → Code_final/
BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = BASE_DIR / "datasets" / "Annotations"
NUSCENES_TRAINVAL_ROOT = BASE_DIR / "datasets" / "nuscenes_data" / "v1.0-trainval_meta"
NUSCENES_TEST_ROOT = BASE_DIR / "datasets" / "nuscenes_data" / "v1.0-test_meta"
OUTPUT_DIR = BASE_DIR / "datasets" / "pre_processed_data"

OBS_LEN = 4   # 4 history frames (2s @ 2Hz)
FUT_LEN = 12  # 12 future frames (6s @ 2Hz)
TTL_LEN = OBS_LEN + 1 + FUT_LEN  # = 17, anchor 单独占一帧


# ══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════════════════
def _clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _scene_name_from_number(scene_number: Any) -> str:
    return f"scene-{int(scene_number):04d}"


def _resolve_csv_files(annotations: Path) -> List[Path]:
    path = Path(annotations)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No doScenes CSV files found under: {annotations}")
    return files


def _load_single_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {}
    for c in df.columns:
        c_norm = c.strip().lower()
        if c_norm == "scene number":
            rename_map[c] = "scene_number"
        elif c_norm == "instruction type":
            rename_map[c] = "instruction_type"
        elif c_norm == "instruction":
            rename_map[c] = "instruction"
    df = df.rename(columns=rename_map)
    required = {"scene_number", "instruction_type", "instruction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    df["scene_number"] = pd.to_numeric(df["scene_number"], errors="coerce")
    df = df.dropna(subset=["scene_number"]).copy()
    df["scene_number"] = df["scene_number"].astype(int)
    df["instruction"] = df["instruction"].map(_clean_text)
    df["instruction_type"] = df["instruction_type"].map(_clean_text)
    df["annotator_file"] = csv_path.name
    df["scene_name"] = df["scene_number"].map(_scene_name_from_number)
    return df


def build_scene_instruction_variants(
    annotations_dir: Path,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    scene_name -> list of (instruction, instruction_type, annotator_file)，
    按 annotator_file 文件名排序后再按行号原顺序保留。仅保留非空 instruction
    （与 ``include_blank_instructions=False`` 一致）。没有指令的 scene 不在
    返回字典中，外部应给它一条空指令 fallback。
    """
    files = _resolve_csv_files(annotations_dir)
    frames = [_load_single_csv(f) for f in files]
    merged = pd.concat(frames, ignore_index=True)

    out: Dict[str, List[Tuple[str, str, str]]] = {}
    for row in merged.itertuples(index=False):
        if not row.instruction:
            continue
        name = row.scene_name
        tup = (row.instruction, row.instruction_type, row.annotator_file)
        out.setdefault(name, []).append(tup)
    # 按 annotator_file 的字典序排序，使下游 "first" 选取可复现
    for name in out:
        out[name].sort(key=lambda t: t[2])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 训练 / 验证：滑动窗口扩样本
# ══════════════════════════════════════════════════════════════════════════════
def _append_sliding_windows(
    records: List[dict],
    scene_name: str,
    scene_token: str,
    item: Dict[str, Any],
    instruction: str,
    instruction_type: str,
    annotator_file: str,
) -> None:
    """
    在 scene 上做 4+1+12 滑动窗口采样。为了贴合官方 leaderboard 协议，每条样本
    都保存 anchor 帧的 (xy, yaw, sample_token)，便于后续把预测从 ego 反投回
    world frame，进而调用官方 ``compute_ego_metrics`` 计算 map 相关指标。
    """
    ego_xy = item["ego_xy"].numpy()
    ego_yaw = item["ego_yaw"].numpy()
    sample_tokens = item["sample_tokens"]

    scene_length = len(ego_xy)
    if scene_length < TTL_LEN:
        return

    ref_flags = _decode_instruction_type(instruction_type)

    # 每条样本的 17 帧布局: [i .. i+OBS_LEN-1] history, [i+OBS_LEN] anchor,
    # [i+OBS_LEN+1 .. i+TTL_LEN-1] future
    for i in range(scene_length - TTL_LEN + 1):
        anchor_idx = i + OBS_LEN  # = i + 4，独立于 history
        anchor_xy_world = ego_xy[anchor_idx]
        anchor_yaw = ego_yaw[anchor_idx]
        if np.isnan(anchor_yaw):
            continue

        c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
        # world -> ego: 先平移再旋转 (-yaw)，等价于把车头方向对齐到 +x
        R_world_to_ego = np.array([[c, s], [-s, c]])

        hist_world = ego_xy[i : i + OBS_LEN]                       # [4, 2]
        fut_world = ego_xy[anchor_idx + 1 : i + TTL_LEN]           # [12, 2]

        hist_ego = (hist_world - anchor_xy_world) @ R_world_to_ego.T
        fut_ego = (fut_world - anchor_xy_world) @ R_world_to_ego.T

        rec = {
            "scene_name": scene_name,
            "scene_token": scene_token,
            # 提交时 sample_token 用 anchor 帧 token（官方 dataloader 也叫
            # anchor_sample_token），保持一致
            "sample_token": sample_tokens[anchor_idx],
            "history_traj": hist_ego.astype(np.float32),
            "future_traj": fut_ego.astype(np.float32),
            # world-frame anchor，反算时使用
            "anchor_xy": anchor_xy_world.astype(np.float32),
            "anchor_yaw": float(anchor_yaw),
            "instruction": instruction,
            "instruction_type": instruction_type,
            "annotator_file": annotator_file,
            **ref_flags,
        }
        records.append(rec)


def process_split(
    nusc: NuScenes,
    split_scenes: set,
    annotations_dir: Path,
    out_path: Path,
) -> None:
    print(f"\n[trainval] 正在处理分割 -> {out_path.name}")
    scene_instr = build_scene_instruction_variants(annotations_dir)
    print(f"  doScenes 有指令的场景数: {len(scene_instr)}")

    scene_by_name = {s["name"]: s for s in nusc.scene}
    records: List[dict] = []
    n_scenes = 0
    n_skipped_short = 0

    for name in sorted(split_scenes):
        scene = scene_by_name.get(name)
        if scene is None:
            continue
        n_scenes += 1
        scene_token = scene["token"]

        if scene.get("nbr_samples", 0) < TTL_LEN:
            n_skipped_short += 1
            continue

        item = load_scene_trajectory(nusc, scene_token, camera_channels=())

        if len(item["ego_xy"]) < TTL_LEN:
            n_skipped_short += 1
            continue

        variants = scene_instr.get(name)
        if not variants:
            _append_sliding_windows(
                records,
                name,
                scene_token,
                item,
                instruction="",
                instruction_type="",
                annotator_file="",
            )
        else:
            for instruction, instruction_type, annotator_file in variants:
                _append_sliding_windows(
                    records,
                    name,
                    scene_token,
                    item,
                    instruction=instruction,
                    instruction_type=instruction_type,
                    annotator_file=annotator_file,
                )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(
        f"[trainval] 保存完毕 -> {out_path}  (场景 {n_scenes}, 过短跳过 "
        f"{n_skipped_short}, 滑动窗口样本 {len(records)})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 测试: 每个 scene 仅 first-segment 1 条样本 (官方 leaderboard 协议)
# ══════════════════════════════════════════════════════════════════════════════
def _build_first_segment_record(
    scene_name: str,
    scene_token: str,
    item: Dict[str, Any],
    instruction: str,
    instruction_type: str,
    annotator_file: str,
) -> Dict[str, Any] | None:
    """
    用 scene 前 17 帧 (i=0) 构造一条样本：history=[0..3], anchor=4,
    future=[5..16] (test 集 future 是缺位的，这里仍按 17 帧约定切，但 future
    在 test 集上不可用——我们 *也* 把它放进去，仅在调试时使用，submit 阶段
    不会读取 future_traj)。
    """
    ego_xy = item["ego_xy"].numpy()
    ego_yaw = item["ego_yaw"].numpy()
    sample_tokens = item["sample_tokens"]

    scene_length = len(ego_xy)
    if scene_length < TTL_LEN:
        return None

    anchor_idx = OBS_LEN  # = 4
    anchor_xy_world = ego_xy[anchor_idx]
    anchor_yaw = ego_yaw[anchor_idx]
    if np.isnan(anchor_yaw):
        return None

    c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
    R_world_to_ego = np.array([[c, s], [-s, c]])

    hist_world = ego_xy[0:OBS_LEN]                                # [4, 2]
    fut_world = ego_xy[anchor_idx + 1 : TTL_LEN]                  # [12, 2]
    hist_ego = (hist_world - anchor_xy_world) @ R_world_to_ego.T
    fut_ego = (fut_world - anchor_xy_world) @ R_world_to_ego.T

    ref_flags = _decode_instruction_type(instruction_type)
    return {
        "scene_name": scene_name,
        "scene_token": scene_token,
        "sample_token": sample_tokens[anchor_idx],
        "history_traj": hist_ego.astype(np.float32),
        "future_traj": fut_ego.astype(np.float32),
        "anchor_xy": anchor_xy_world.astype(np.float32),
        "anchor_yaw": float(anchor_yaw),
        "instruction": instruction,
        "instruction_type": instruction_type,
        "annotator_file": annotator_file,
        **ref_flags,
    }


def process_test_split(
    nusc: NuScenes,
    test_scenes: set,
    annotations_dir: Path,
    out_path: Path,
) -> None:
    """
    生成 test pkl: 每个 test scene 仅 1 条样本，使用 scene 前 17 帧；同 scene
    多条 instruction 时取按 annotator_file 字典序排序后的第一条非空 instruction
    （等价于 dataset.py 的 ``instruction_mode='first'``，确定性可复现）。
    """
    print(f"\n[test] 正在处理 test 集 -> {out_path.name}")
    scene_instr = build_scene_instruction_variants(annotations_dir)
    print(f"  doScenes 有指令的场景数 (含非 test): {len(scene_instr)}")

    scene_by_name = {s["name"]: s for s in nusc.scene}
    records: List[dict] = []
    n_scenes = 0
    n_skipped_short = 0
    n_no_instr = 0

    for name in sorted(test_scenes):
        scene = scene_by_name.get(name)
        if scene is None:
            continue
        n_scenes += 1
        scene_token = scene["token"]

        if scene.get("nbr_samples", 0) < TTL_LEN:
            n_skipped_short += 1
            continue

        item = load_scene_trajectory(nusc, scene_token, camera_channels=())

        if len(item["ego_xy"]) < TTL_LEN:
            n_skipped_short += 1
            continue

        variants = scene_instr.get(name)
        if variants:
            instruction, instruction_type, annotator_file = variants[0]
        else:
            n_no_instr += 1
            instruction, instruction_type, annotator_file = "", "", ""

        rec = _build_first_segment_record(
            scene_name=name,
            scene_token=scene_token,
            item=item,
            instruction=instruction,
            instruction_type=instruction_type,
            annotator_file=annotator_file,
        )
        if rec is None:
            n_skipped_short += 1
            continue
        records.append(rec)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(
        f"[test] 保存完毕 -> {out_path}  (test 场景 {n_scenes}, 过短跳过 "
        f"{n_skipped_short}, 无 instruction {n_no_instr}, 写出样本 {len(records)})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess doScenes Track-2 pkl files")
    p.add_argument("--annotations_dir", default=str(ANNOTATION_DIR),
                   help="doScenes Annotations 目录 (含若干 annotator csv)")
    p.add_argument("--nuscenes_trainval_root", default=str(NUSCENES_TRAINVAL_ROOT),
                   help="NuScenes v1.0-trainval dataroot (含 v1.0-trainval/ 子目录)")
    p.add_argument("--nuscenes_test_root", default=str(NUSCENES_TEST_ROOT),
                   help="NuScenes v1.0-test dataroot (含 v1.0-test/ 子目录)")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR),
                   help="输出 pkl 目录")
    p.add_argument("--skip_train_val", action="store_true",
                   help="跳过 train/val pkl 生成")
    p.add_argument("--skip_test", action="store_true",
                   help="跳过 test pkl 生成")
    return p.parse_args()


def main() -> None:
    args = get_args()
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 让 OUTPUT_DIR 也跟随命令行参数
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir

    splits = create_splits_scenes()

    if not args.skip_train_val:
        print("\n========== Loading NuScenes v1.0-trainval ==========")
        nusc_tv = NuScenes(version="v1.0-trainval",
                           dataroot=str(args.nuscenes_trainval_root), verbose=True)
        process_split(nusc_tv, set(splits["train"]), annotations_dir,
                      output_dir / "train_track2.pkl")
        process_split(nusc_tv, set(splits["val"]), annotations_dir,
                      output_dir / "val_track2.pkl")
        del nusc_tv
    else:
        print("[main] --skip_train_val 已开启，跳过 train/val pkl 生成")

    if not args.skip_test:
        print("\n========== Loading NuScenes v1.0-test ==========")
        nusc_te = NuScenes(version="v1.0-test",
                           dataroot=str(args.nuscenes_test_root), verbose=True)
        process_test_split(nusc_te, set(splits["test"]), annotations_dir,
                           output_dir / "test_track2.pkl")
        del nusc_te
    else:
        print("[main] --skip_test 已开启，跳过 test pkl 生成")

    print("\n[done] 全部预处理完成。")


if __name__ == "__main__":
    main()
