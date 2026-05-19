"""
velocity_net.py
===============
Flow Matching velocity field network for doScenes Track-2.

Architecture overview
---------------------
Given the noisy trajectory x_t, flow time t, and the conditional context c,
the network predicts the velocity field v(x_t, t, c) ≈ x_1 - x_0.

Dual-guidance pipeline
----------------------
Inspired by ref_decoder.py ConcatSquashLinear, each layer applies two
sequential STFT-style gated conditions (FiLM gate + bias) before passing
through a Transformer sub-layer:

  x_t + t_emb + pos_emb
        │
  ┌─────▼──────────────────────────────────────────┐
  │  DualGuidanceBlock (repeated n_layers times)   │
  │                                                 │
  │  GatedConditionBlock(c_hist, x) → Transformer  │  ← hist guidance
  │  + residual skip                                │
  │                                                 │
  │  GatedConditionBlock(c_lang, x) → Transformer  │  ← lang guidance
  │  + residual skip                                │
  └─────────────────────────────────────────────────┘
        │
  out_proj  Linear(embed_dim → 2)
        │
  v_pred  [B, T_fut, 2]

The cleaned training pipeline always uses this dual-guidance path.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Sinusoidal Time Embedding
# ══════════════════════════════════════════════════════════════════════════════
class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps a scalar flow-time t ∈ [0, 1] to a dense embedding vector.

    Architecture
    ------------
    Classic sinusoidal encoding (Vaswani et al., 2017) adapted for a
    continuous scalar:

        half_dim frequencies   f_k = exp(-log(max_period) * k / half_dim)
        raw embedding          [sin(t * f_0), …, cos(t * f_{half_dim-1})]
        → shape [B, 2 * half_dim]

    Followed by a two-layer MLP to project to the final `embed_dim`,
    letting the network learn a non-linear remapping of time.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.  Must be even.
    max_period : float
        Controls the minimum frequency.  10 000 (the Transformer default)
        gives fine resolution over t ∈ [0, 1].
    """

    def __init__(self, embed_dim: int = 128, max_period: float = 10_000.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal encoding"
        self.embed_dim  = embed_dim
        self.half_dim   = embed_dim // 2
        self.max_period = max_period

        # Learnable MLP refines the raw sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),                            # Swish activation
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : FloatTensor [B]   values in [0, 1]

        Returns
        -------
        FloatTensor [B, embed_dim]
        """
        # Compute frequencies: [half_dim]
        half  = self.half_dim
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        # Outer product: [B, half_dim]
        args = t[:, None].float() * freqs[None, :]

        # Sinusoidal features: [B, embed_dim]
        raw_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(raw_emb)   # [B, embed_dim]


# ══════════════════════════════════════════════════════════════════════════════
# Gated Condition Block  (equivalent to traj_STFT / image_STFT in ref_decoder)
# ══════════════════════════════════════════════════════════════════════════════
class GatedConditionBlock(nn.Module):
    """
    FiLM-style gated conditioning: fuses a context vector into a sequence.

    Mirrors the STFT blocks in ref_decoder.py (traj_STFT / image_STFT).
    For each trajectory timestep the block computes a gate and a bias from the
    context vector `ctx` and the current sequence representation `x`, then
    applies:

        out = x * sigmoid(gate) + bias

    Parameters
    ----------
    dim_x   : int   dimension of the sequence features (embed_dim)
    dim_ctx : int   dimension of the condition context vector (hidden_dim)
    """

    def __init__(self, dim_x: int, dim_ctx: int) -> None:
        super().__init__()
        # Projects the sequence features
        self.input_layer   = nn.Linear(dim_x,   dim_x)
        # Projects the context vector (broadcast over T)
        self.context_layer = nn.Linear(dim_ctx,  dim_x)
        # Gate and bias computed from concatenated [seq_feat, ctx_feat]
        self.hyper_gate    = nn.Linear(dim_x * 2, dim_x)
        self.hyper_bias    = nn.Linear(dim_x * 2, dim_x, bias=False)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ctx : FloatTensor [B, dim_ctx]   condition vector (history or language)
        x   : FloatTensor [B, T, dim_x] sequence to be conditioned

        Returns
        -------
        FloatTensor [B, T, dim_x]
        """
        emb_x   = self.input_layer(x)                              # [B, T, D]
        emb_ctx = self.context_layer(ctx)                          # [B, D]
        # Expand context across timesteps then concatenate
        emb_cat = torch.cat(
            [emb_x, emb_ctx.unsqueeze(1).expand_as(emb_x)], dim=-1
        )                                                           # [B, T, 2D]
        gate = torch.sigmoid(self.hyper_gate(emb_cat))            # [B, T, D]
        bias = self.hyper_bias(emb_cat)                            # [B, T, D]
        return emb_x * gate + bias                                 # [B, T, D]


# ══════════════════════════════════════════════════════════════════════════════
# Dual Guidance Block  (history guidance → lang guidance, each with Transformer)
# ══════════════════════════════════════════════════════════════════════════════
class DualGuidanceBlock(nn.Module):
    """
    One dual-guidance decoder layer, mirroring ConcatSquashLinear in ref_decoder.

    Internal flow
    -------------
    input x  [B, T, D]
      │
      ├─ GatedConditionBlock(c_hist, x)   → x_hist [B, T, D]
      │  TransformerEncoderLayer(x_hist)  → trans1 [B, T, D]
      │  ret1 = residual_proj(x) + trans1            ← skip connection
      │
      └─ GatedConditionBlock(c_lang, ret1) → x_lang [B, T, D]
         TransformerEncoderLayer(x_lang)  → trans2 [B, T, D]
         output = ret1 + trans2                       ← skip connection

    Parameters
    ----------
    embed_dim : int   model width (D)
    n_heads   : int   attention heads inside each Transformer sub-layer
    dropout   : float dropout rate
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        # ── Gated condition blocks ────────────────────────────────────────────
        self.hist_gate = GatedConditionBlock(dim_x=embed_dim, dim_ctx=embed_dim)
        self.lang_gate = GatedConditionBlock(dim_x=embed_dim, dim_ctx=embed_dim)

        # ── Transformer sub-layers (one per guidance branch) ─────────────────
        tf_kwargs = dict(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tf_hist = nn.TransformerEncoderLayer(**tf_kwargs)
        self.tf_lang = nn.TransformerEncoderLayer(**tf_kwargs)

        # ── Skip-connection projection for the first branch ───────────────────
        # Projects the original input x to embed_dim (identity if dims match)
        self.residual_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x:      torch.Tensor,  # [B, T, embed_dim]
        c_hist: torch.Tensor,  # [B, embed_dim]  history context
        c_lang: torch.Tensor,  # [B, embed_dim]  language context
    ) -> torch.Tensor:
        # ── Branch 1: history guidance ────────────────────────────────────────
        x_hist = self.hist_gate(c_hist, x)           # [B, T, D]
        trans1 = self.tf_hist(x_hist)                # [B, T, D]
        ret1   = self.residual_proj(x) + trans1      # [B, T, D]  residual

        # ── Branch 2: language guidance ───────────────────────────────────────
        x_lang = self.lang_gate(c_lang, ret1)        # [B, T, D]
        trans2 = self.tf_lang(x_lang)                # [B, T, D]
        output = ret1 + trans2                       # [B, T, D]  residual

        return output


# ══════════════════════════════════════════════════════════════════════════════
# Velocity Field Network
# ══════════════════════════════════════════════════════════════════════════════
class VelocityNetwork(nn.Module):
    """
    Predicts the flow-matching velocity field v(x_t, t, c).

    Parameters
    ----------
    traj_dim : int
        Spatial dimension of each trajectory point (default 2 for x, y).
    fut_len : int
        Number of future time steps T_fut (default 12).
    condition_dim : int
        Dimensionality of the context vector c from ConditionEncoder.
    embed_dim : int
        Internal model width.  All sub-modules share this dimension.
    n_heads : int
        Number of Transformer attention heads.  embed_dim % n_heads must == 0.
    n_layers : int
        Number of DualGuidanceBlocks. Each block contains two Transformer
        sub-layers.
    dropout : float
        Dropout rate inside the Transformer and projection heads.
    c : tuple
        The forward call expects (c_hist [B, condition_dim], c_lang [B,
        condition_dim]).
    """

    def __init__(
        self,
        traj_dim:          int   = 2,
        fut_len:           int   = 12,
        condition_dim:     int   = 128,
        embed_dim:         int   = 256,
        n_heads:           int   = 4,
        n_layers:          int   = 3,
        dropout:           float = 0.1,
    ) -> None:
        super().__init__()
        self.fut_len   = fut_len
        self.embed_dim = embed_dim

        # ── 1. Input projection: trajectory points → embed_dim ──────────────
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── 2. Learned positional embedding for the T_fut trajectory steps ──
        self.pos_emb = nn.Embedding(fut_len, embed_dim)

        # ── 3. Time embedding ────────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(embed_dim=embed_dim)

        # ── 4. Dual-guidance projections ─────────────────────────────────────
        self.hist_proj = nn.Sequential(
            nn.Linear(condition_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.lang_proj = nn.Sequential(
            nn.Linear(condition_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── 5. Dual-guidance decoder blocks ──────────────────────────────────
        self.dual_blocks = nn.ModuleList([
            DualGuidanceBlock(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.dual_norm = nn.LayerNorm(embed_dim)

        # ── 6. Output projection: embed_dim → traj_dim ───────────────────────
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, traj_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize output projection near zero for training stability."""
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(
        self,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        c:        Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        """
        Parameters
        ----------
        x_t      : FloatTensor [B, T_fut, 2]
            Linearly interpolated (noisy) trajectory at flow time t.
        t        : FloatTensor [B]
            Continuous flow time in [0, 1].
        c        : Tuple of two tensors
            Dual-guidance context (c_hist, c_lang), each [B, condition_dim],
            where c_hist carries history context and c_lang carries language context.

        Returns
        -------
        v_pred       : FloatTensor [B, T_fut, 2]
            Predicted velocity field (target: x_1 - x_0).
        None         : kept for compatibility with existing call sites that
            unpack ``v_pred, _``.
        """
        B, T, _ = x_t.shape

        # ── Step 1: project trajectory to embed space ────────────────────────
        h = self.traj_proj(x_t)                       # [B, T, embed_dim]

        # ── Step 2: add learned positional encoding ──────────────────────────
        positions = torch.arange(T, device=x_t.device)
        h = h + self.pos_emb(positions)               # [B, T, embed_dim]

        # ── Step 3: inject time embedding (broadcast over T) ─────────────────
        t_emb = self.time_emb(t)                      # [B, embed_dim]
        h = h + t_emb.unsqueeze(1)                    # [B, T, embed_dim]

        # ── Step 4: project both conditions then pass through dual blocks ────
        c_hist, c_lang = c
        c_hist_emb = self.hist_proj(c_hist)       # [B, embed_dim]
        c_lang_emb = self.lang_proj(c_lang)       # [B, embed_dim]

        for block in self.dual_blocks:
            h = block(h, c_hist_emb, c_lang_emb)  # [B, T, embed_dim]

        h = self.dual_norm(h)                     # [B, T, embed_dim]

        # ── Step 5: predict velocity at each future step ─────────────────────
        v_pred = self.out_proj(h)                      # [B, T, 2]

        return v_pred, None