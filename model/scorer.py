"""
scorer.py
=========
Trajectory Scorer for doScenes Track-2 Flow Matching model.

Role in the pipeline
--------------------
After the Flow Matching model generates K *correlated* candidate trajectories
(via FlowMatchingTrainer.sample_correlated), the scorer ranks them and returns
the one most likely to match the ground-truth future.

Architecture
------------
  CandidateEncoder   — lightweight BiGRU encodes each future trajectory
                       [B, K, T_fut, 2]  →  [B, K, hidden_dim]

  TrajectoryScorer   — cross-attention: K candidates (Q) attend to the
                       shared condition context c (K, V), then a scoring
                       MLP outputs one scalar per candidate.
                       [B, hidden_dim] + [B, K, T_fut, 2]  →  [B, K]

Training objective  (scorer_loss)
----------------------------------
Given K generated candidates and the GT future trajectory, compute the
*true* ADE for each candidate.  Convert ADE values to a soft target
distribution via softmax(-ADE / τ) — the best candidate (lowest ADE)
receives the highest target probability.

Loss = cross-entropy( softmax(predicted_scores),  softmax(-ADE/τ) )
     ≡ KL( soft_target  ||  softmax(pred_scores) )   + constant

This formulation is:
  • Scale-invariant  — works regardless of absolute ADE magnitudes.
  • Ranking-aware    — pushes the scorer to distinguish good from bad.
  • Smooth           — temperature τ controls target sharpness.

Usage at inference
------------------
  candidates     = model.sample_correlated(history, instructions, n_samples=K)
  c              = model.condition_encoder(history, instructions)
  best_traj, _   = scorer.select_best(c, candidates)

Dual-guidance context
---------------------
The cleaned Flow Matching model returns a tuple (c_hist, c_lang). The scorer
handles this via _to_kv(), so candidates attend to both history and language
context independently in the cross-attention.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. CandidateEncoder
# ══════════════════════════════════════════════════════════════════════════════
class CandidateEncoder(nn.Module):
    """
    Encodes a future trajectory into a fixed-size feature vector.

    Architecture
    ------------
    Linear input projection  →  1-layer BiGRU  →  attention pooling

    Deliberately lightweight: it will be called on B*K trajectories per step,
    so we keep n_layers=1 and no residual stack.

    Parameters
    ----------
    traj_dim   : spatial dimension of each waypoint (default 2 for x, y)
    hidden_dim : output feature dimension
    """

    def __init__(self, traj_dim: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_pool = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        traj : FloatTensor [..., T_fut, traj_dim]
            Any number of leading batch dimensions are supported.

        Returns
        -------
        FloatTensor [..., hidden_dim]
        """
        *batch_dims, T, D = traj.shape
        x = traj.reshape(-1, T, D)                  # [N, T, D]

        x   = self.input_proj(x)                    # [N, T, hidden_dim]
        out, _ = self.gru(x)                        # [N, T, 2*hidden_dim]
        seq = self.out_proj(out)                    # [N, T, hidden_dim]

        scores  = self.attn_pool(seq)               # [N, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled  = (weights * seq).sum(dim=1)        # [N, hidden_dim]

        return pooled.reshape(*batch_dims, -1)       # [..., hidden_dim]

    def encode_frames(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Return per-timestep BiGRU features without attention pooling.

        Parameters
        ----------
        traj : FloatTensor [..., T_fut, traj_dim]

        Returns
        -------
        FloatTensor [..., T_fut, hidden_dim]
        """
        *batch_dims, T, D = traj.shape
        x = traj.reshape(-1, T, D)          # [N, T, D]
        x = self.input_proj(x)              # [N, T, hidden_dim]
        out, _ = self.gru(x)               # [N, T, 2*hidden_dim]
        seq = self.out_proj(out)            # [N, T, hidden_dim]
        return seq.reshape(*batch_dims, T, -1)  # [..., T, hidden_dim]


# ══════════════════════════════════════════════════════════════════════════════
# 2. TrajectoryScorer
# ══════════════════════════════════════════════════════════════════════════════
class TrajectoryScorer(nn.Module):
    """
    Ranks K candidate future trajectories given the condition context c.

    The scorer takes the pre-computed condition vector from ConditionEncoder
    (which already encodes both history and language instruction) so that
    DistilBERT does not need to run twice.

    Architecture (forward pass)
    ---------------------------
    1. CandidateEncoder  :  [B, K, T_fut, 2]  →  cand_feat  [B, K, D]
    2. Cross-attention   :  candidates (Q) attend to context c (K, V)
                            → enriches each candidate's feature with the
                               scene / instruction context
       Residual + LayerNorm keeps the gradient flow healthy.
    3. Scoring MLP       :  [B, K, D]  →  scores  [B, K]

    Dual-guidance support
    ---------------------
    When c is a tuple (c_hist, c_lang), _to_kv() stacks them into [B, 2, D]
    so that the cross-attention keys/values span both signals.  No extra
    parameters are required.

    Parameters
    ----------
    hidden_dim : must match ConditionEncoder.hidden_dim  (default 128)
    traj_dim   : trajectory spatial dim                  (default 2)
    fut_len    : future timesteps                        (default 12)
    n_heads    : cross-attention heads
    dropout    : dropout in cross-attention and scoring MLP
    """

    def __init__(
        self,
        hidden_dim: int   = 128,
        traj_dim:   int   = 2,
        fut_len:    int   = 12,
        n_heads:    int   = 4,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.candidate_enc = CandidateEncoder(
            traj_dim=traj_dim, hidden_dim=hidden_dim
        )

        # Cross-attention: K candidates (Q) attend to context c (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Scoring MLP: feature → scalar
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the final linear near zero for stable early training."""
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    @staticmethod
    def _to_kv(c: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Convert a condition to a key/value tensor for cross-attention.

        Single-guidance : c is FloatTensor [B, D]
                          → returns [B, 1, D]
        Dual-guidance   : c is (c_hist [B,D], c_lang [B,D])
                          → stacks to [B, 2, D] so candidates attend to
                            both history and language context independently.
        """
        if isinstance(c, tuple):
            return torch.stack(list(c), dim=1)   # [B, 2, D]
        return c.unsqueeze(1)                    # [B, 1, D]

    def forward(
        self,
        c: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        candidates: torch.Tensor,               # [B, K, T_fut, traj_dim]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        c          : condition context from ConditionEncoder.
                     FloatTensor [B, hidden_dim]  (single-guidance)  OR
                     Tuple (c_hist, c_lang) each [B, hidden_dim]     (dual-guidance).
        candidates : K candidate future trajectories  [B, K, T_fut, traj_dim]

        Returns
        -------
        scores : FloatTensor [B, K]
            Raw logits (not softmaxed). Higher = better quality.
        """
        # ── 1. Encode each candidate ─────────────────────────────────────────
        cand_feat = self.candidate_enc(candidates)      # [B, K, hidden_dim]

        # ── 2. Cross-attention: each candidate attends to the context ─────────
        # c_kv: [B, 1, D] for single-guidance or [B, 2, D] for dual-guidance
        c_kv = self._to_kv(c)
        attn_out, _ = self.cross_attn(
            query=cand_feat,   # Q: [B, K, D]
            key=c_kv,          # K: [B, 1/2, D]
            value=c_kv,        # V: [B, 1/2, D]
        )                                               # [B, K, D]
        cand_feat = self.norm(cand_feat + attn_out)     # residual connection

        # ── 3. Score each candidate ───────────────────────────────────────────
        scores = self.score_head(cand_feat).squeeze(-1)  # [B, K]
        return scores

    def select_best(
        self,
        c: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        candidates: torch.Tensor,               # [B, K, T_fut, traj_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Score all candidates and return the highest-scoring one per scene.

        Returns
        -------
        best_traj : FloatTensor [B, T_fut, traj_dim]
        scores    : FloatTensor [B, K]   (raw logits for all candidates)
        """
        scores   = self.forward(c, candidates)      # [B, K]
        best_idx = scores.argmax(dim=-1)            # [B]
        best_traj = candidates[
            torch.arange(candidates.size(0), device=candidates.device),
            best_idx,
        ]                                           # [B, T_fut, traj_dim]
        return best_traj, scores

    def select_framewise(
        self,
        c: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        candidates: torch.Tensor,               # [B, K, T_fut, traj_dim]
    ) -> torch.Tensor:
        """
        Greedy frame-by-frame selection: at each timestep independently pick
        the candidate point with the highest score, then assemble into a full
        trajectory.

        This reuses the trained cross_attn, norm, and score_head weights but
        applies them at per-frame granularity rather than trajectory level.

        Returns
        -------
        assembled : FloatTensor [B, T_fut, traj_dim]
        """
        B, K, T, traj_dim = candidates.shape

        # Step 1: per-frame BiGRU features for every candidate — [B, K, T, D]
        frame_feats = self.candidate_enc.encode_frames(candidates)

        # Step 2: reshape so each timestep is processed independently
        #   [B, K, T, D] → permute → [B, T, K, D] → [B*T, K, D]
        frame_feats = frame_feats.permute(0, 2, 1, 3).reshape(B * T, K, -1)

        # Step 3: expand condition to match each (batch, timestep) pair
        # _to_kv gives [B, 1, D] or [B, 2, D]; expand T times → [B*T, 1/2, D]
        c_base = self._to_kv(c)                                    # [B, N_kv, D]
        N_kv   = c_base.size(1)
        c_kv   = (
            c_base.unsqueeze(1)                                    # [B, 1, N_kv, D]
            .expand(-1, T, -1, -1)                                 # [B, T, N_kv, D]
            .reshape(B * T, N_kv, -1)                              # [B*T, N_kv, D]
        )

        # Step 4: cross-attention — each frame's K candidates attend to context
        attn_out, _ = self.cross_attn(
            query=frame_feats,  # [B*T, K, D]
            key=c_kv,           # [B*T, N_kv, D]
            value=c_kv,
        )
        frame_feats = self.norm(frame_feats + attn_out)   # [B*T, K, D]

        # Step 5: score each candidate at each frame → [B, T, K]
        frame_scores = self.score_head(frame_feats).squeeze(-1)  # [B*T, K]
        frame_scores = frame_scores.reshape(B, T, K)

        # Step 6: per-frame argmax — which candidate wins at each timestep
        best_k_per_frame = frame_scores.argmax(dim=-1)             # [B, T]

        # Step 7: gather the winning point at each frame from candidates
        B_idx = torch.arange(B, device=candidates.device).unsqueeze(1).expand(-1, T)
        T_idx = torch.arange(T, device=candidates.device).unsqueeze(0).expand(B, -1)
        assembled = candidates[B_idx, best_k_per_frame, T_idx]    # [B, T, traj_dim]

        return assembled


# ══════════════════════════════════════════════════════════════════════════════
# 3. Scorer Loss
# ══════════════════════════════════════════════════════════════════════════════
def scorer_loss(
    pred_scores: torch.Tensor,   # [B, K]  raw scorer output (logits)
    candidates:  torch.Tensor,   # [B, K, T_fut, 2]
    gt_future:   torch.Tensor,   # [B, T_fut, 2]
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Soft-target cross-entropy loss for trajectory scoring.

    Algorithm
    ---------
    1. Compute the true ADE of each candidate against the GT future:
           ADE_k = mean_t ‖ candidate_k[t] − gt[t] ‖₂

    2. Convert ADE values to a soft target distribution:
           target_k = softmax( −ADE_k / τ )
       The best candidate (lowest ADE) receives the highest target probability.
       Temperature τ controls sharpness:
         τ → 0  : hard assignment — only the best candidate has non-zero target
         τ → ∞  : uniform — all candidates are treated equally

    3. Loss = cross-entropy( log_softmax(pred_scores), target_probs )
            ≡ KL( target_probs  ‖  softmax(pred_scores) )  + constant

    Parameters
    ----------
    pred_scores : raw scorer logits       [B, K]
    candidates  : K generated trajectories [B, K, T_fut, 2]
    gt_future   : ground-truth future      [B, T_fut, 2]
    temperature : τ for the soft target.  Default 0.5 gives a moderately
                  peaked distribution — the best ~3-5 out of K=20 get
                  most of the probability mass.

    Returns
    -------
    scalar loss tensor
    """
    # ── Compute true ADE for each candidate ──────────────────────────────────
    diff = candidates.detach() - gt_future.unsqueeze(1)   # [B, K, T_fut, 2]
    ade  = diff.norm(dim=-1).mean(dim=-1)                 # [B, K]

    # ── Soft target distribution: low ADE → high target probability ──────────
    target_probs = torch.softmax(-ade / temperature, dim=-1)   # [B, K]

    # ── Cross-entropy with soft targets ──────────────────────────────────────
    log_pred = F.log_softmax(pred_scores, dim=-1)              # [B, K]
    loss = -(target_probs * log_pred).sum(dim=-1).mean()

    return loss