"""
flow_matching.py
================
Optimal Transport Conditional Flow Matching (OT-CFM) wrapper for
doScenes Track-2 (Language + History) trajectory prediction.

Theory recap
------------
Conditional Flow Matching (Lipman et al., 2022; Liu et al., 2022) defines a
probability path p_t between a source distribution p_0 (standard Gaussian)
and a target distribution p_1 (ground-truth trajectories).

For the Optimal Transport variant, the path between a *paired* (x_0, x_1) is
simply the straight line:

    x_t  = (1 - t) · x_0  +  t · x_1            (linear interpolation)
    v*   =  x_1 - x_0                            (constant target velocity)

The training objective is:

    L = E_{t, x_0, x_1} [ ‖ v_θ(x_t, t, c)  −  (x_1 − x_0) ‖² ]

where c is the condition vector from ConditionEncoder.

At inference, we integrate the learned ODE from x_0 ~ N(0, I) to obtain
a predicted trajectory x_1 via an ODE solver (e.g. Euler, RK4).

File structure
--------------
  FlowMatchingTrainer   — training wrapper (forward = loss computation)
  FlowMatchingInference — inference wrapper (ODE integration)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ConditionEncoder
from .velocity_net import VelocityNetwork


# ══════════════════════════════════════════════════════════════════════════════
# Model Configuration
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class FlowMatchingConfig:
    """
    Single place to tune every hyper-parameter for the entire model.

    Trajectory / data
    -----------------
    obs_len      : history frames  (must match dataset.py OBS_LEN = 4)
    fut_len      : future frames   (must match dataset.py FUT_LEN = 12)
    traj_dim     : spatial dim per step (2 for x, y)

    Condition encoder (ConditionEncoder)
    ------------------------------------
    hidden_dim       : shared feature dimension across all sub-modules
    n_hist_layers    : GRU layers in HistoryEncoder
    n_unfreeze_bert  : DistilBERT blocks to fine-tune from top
    n_heads_encoder  : cross-attention heads in FusionModule
    dropout_encoder  : dropout for encoder stack

    Velocity network (VelocityNetwork)
    -----------------------------------
    embed_dim        : internal width of VelocityNetwork
    n_heads_velocity : attention heads in Transformer stack
    n_layers_velocity: number of TransformerEncoderLayers
    dropout_velocity : dropout for velocity network

    Flow matching
    -------------
    sigma_min        : minimum noise std for OT-CFM.
                       A small non-zero value stabilises training near t=0/1.
                       Typical: 1e-4 – 1e-3.
    t_eps            : clamp t away from exact 0 and 1 to avoid numerical issues.

    Training samples flow time uniformly from [t_eps, 1 - t_eps].
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    obs_len:   int = 4
    fut_len:   int = 12
    traj_dim:  int = 2

    # ── Condition encoder ─────────────────────────────────────────────────────
    hidden_dim:       int   = 128
    n_hist_layers:    int   = 2
    n_unfreeze_bert:  int   = 2
    n_heads_encoder:  int   = 4
    dropout_encoder:  float = 0.1
    max_text_length:  int   = 64

    # ── Velocity network ──────────────────────────────────────────────────────
    embed_dim:         int   = 256
    n_heads_velocity:  int   = 4
    n_layers_velocity: int   = 3
    dropout_velocity:  float = 0.1

    # ── Flow matching ─────────────────────────────────────────────────────────
    sigma_min: float = 1e-4
    t_eps:     float = 1e-4

    # ── Classifier-Free Guidance (CFG) ────────────────────────────────────────
    # cfg_dropout_prob:
    #   Probability of replacing each instruction with "" during training.
    #   This teaches the model both conditioned (p_θ(x|c)) and unconditioned
    #   (p_θ(x)) generation from a single set of weights.
    #   Set to 0.0 to disable CFG entirely (ablation: language-only baseline).
    #   Recommended range: 0.10 – 0.20.
    #
    # cfg_guidance_scale:
    #   Guidance weight w at inference time.
    #   w = 1.0  →  standard conditional sampling  (CFG disabled)
    #   w > 1.0  →  v_guided = v_uncond + w*(v_cond - v_uncond)
    #              amplifies instruction conditioning at the cost of diversity.
    #   Recommended range: 1.0 – 3.0.  Only active when cfg_dropout_prob > 0.
    cfg_dropout_prob:   float = 0.15
    cfg_guidance_scale: float = 1.5

    # ── Dual-Guidance Velocity Network ────────────────────────────────────────
    # ConditionEncoder returns (c_hist [B,D], c_lang [B,D]); each
    # DualGuidanceBlock injects history context then language context.

# ══════════════════════════════════════════════════════════════════════════════
# Flow Matching Trainer  (training-time forward = loss)
# ══════════════════════════════════════════════════════════════════════════════
class FlowMatchingTrainer(nn.Module):
    """
    Wraps ConditionEncoder + VelocityNetwork and exposes a training-oriented
    forward pass that directly returns the OT-CFM loss.

    forward() performs the following steps
    ---------------------------------------
    1.  Encode conditions  →  c  [B, hidden_dim]
    2.  Sample source noise  x_0 ~ N(0, I),  shape [B, T_fut, traj_dim]
    3.  Sample flow time     t   (Uniform or Logit-Normal → (0,1), see cfg), [B]
    4.  Linear interpolation x_t = (1-t) · x_0 + t · x_1   (OT path)
    5.  Target velocity      v*  = x_1 - x_0               (constant along OT path)
    6.  Predict              v_pred = VelocityNetwork(x_t, t_flat, c)
    7.  Loss                 L = MSE(v_pred, v*)

    Parameters
    ----------
    cfg : FlowMatchingConfig
        All hyper-parameters for every sub-module.
    """

    def __init__(self, cfg: Optional[FlowMatchingConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or FlowMatchingConfig()
        c = self.cfg   # short alias

        self.condition_encoder = ConditionEncoder(
            hidden_dim=c.hidden_dim,
            d_in=c.traj_dim,
            n_hist_layers=c.n_hist_layers,
            n_unfreeze_bert=c.n_unfreeze_bert,
            n_heads=c.n_heads_encoder,
            dropout=c.dropout_encoder,
            max_length=c.max_text_length,
            obs_len=c.obs_len,
        )
        self.velocity_net = VelocityNetwork(
            traj_dim=c.traj_dim,
            fut_len=c.fut_len,
            condition_dim=c.hidden_dim,
            embed_dim=c.embed_dim,
            n_heads=c.n_heads_velocity,
            n_layers=c.n_layers_velocity,
            dropout=c.dropout_velocity,
        )

    # ── OT-CFM sample construction ────────────────────────────────────────────
    def _sample_ot_cfm(
        self,
        x_1: torch.Tensor,    # [B, T_fut, traj_dim]  ground-truth future
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs one OT-CFM sample:

        Returns
        -------
        x_0       : sampled Gaussian noise        [B, T_fut, traj_dim]
        t_scalar  : flow time per sample          [B]
        x_t       : interpolated trajectory       [B, T_fut, traj_dim]
        v_target  : target velocity               [B, T_fut, traj_dim]
        """
        B = x_1.size(0)
        sigma = self.cfg.sigma_min
        eps   = self.cfg.t_eps

        # Step 1 — sample source noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)

        # Step 2 — sample continuous time t uniformly in (eps, 1 - eps)
        t_scalar = torch.rand(B, device=x_1.device, dtype=x_1.dtype) * (
            1 - 2 * eps
        ) + eps
        t_bc = t_scalar.view(B, 1, 1)          # broadcast shape

        # Step 3 — OT linear interpolation with optional sigma_min noise floor
        #   x_t = (1 - t) * x_0  +  t * x_1
        #   (sigma_min nudges x_t slightly off the exact OT line, which can
        #    improve gradient conditioning near t=0 and t=1)
        mu_t    = (1.0 - t_bc) * x_0 + t_bc * x_1
        x_t     = mu_t + sigma * torch.randn_like(x_1)  # small noise floor

        # Step 4 — constant target velocity along the OT path
        v_target = x_1 - x_0                           # [B, T_fut, traj_dim]

        return x_0, t_scalar, x_t, v_target

    # ── Training forward pass ─────────────────────────────────────────────────
    def forward(
        self,
        x_1:          torch.Tensor,  # [B, T_fut, 2]   ground-truth future
        history:      torch.Tensor,  # [B, T_hist, 2]  observed past
        instructions: List[str],     # length B
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the OT-CFM training loss.

        Parameters
        ----------
        x_1          : ground-truth future trajectory [B, T_fut, 2]
        history      : observed past trajectory       [B, T_hist, 2]
        instructions : list of B instruction strings

        Returns
        -------
        dict with keys:
            "loss"     : scalar MSE loss
            "v_pred"   : predicted velocity  [B, T_fut, 2]  (for diagnostics)
            "v_target" : target velocity     [B, T_fut, 2]  (for diagnostics)
        """
        # ── 1. CFG instruction dropout (training only) ───────────────────────
        # With probability cfg_dropout_prob, replace each instruction with the
        # null string so the model learns unconditional generation as well.
        # When cfg_dropout_prob == 0.0 this block is skipped entirely, giving
        # a standard conditional model (useful for the ablation baseline).
        if self.training and self.cfg.cfg_dropout_prob > 0.0:
            drop_mask  = torch.rand(len(instructions)) < self.cfg.cfg_dropout_prob
            instructions = [
                "" if drop else instr
                for drop, instr in zip(drop_mask.tolist(), instructions)
            ]

        # ── 2. Encode conditions ─────────────────────────────────────────────
        c = self.condition_encoder(history, instructions)   # [B, hidden_dim]

        # ── 3 / 4 / 5. Standard single-mode OT-CFM objective ───────────────
        _, t_scalar, x_t, v_target = self._sample_ot_cfm(x_1)
        v_pred, _ = self.velocity_net(x_t, t_scalar, c)       # [B, T_fut, 2]
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss":     loss,
            "v_pred":   v_pred.detach(),
            "v_target": v_target.detach(),
        }

    # ── Euler ODE Sampler (inference) ─────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        history:        torch.Tensor,   # [B, T_hist, 2]
        instructions:   List[str],      # length B
        n_steps:        int   = 50,
        n_samples:      int   = 1,
        guidance_scale: Optional[float] = None,
        return_traj:    bool  = False,
    ) -> torch.Tensor:
        """
        Generate predicted future trajectories via Euler-method ODE integration,
        with optional Classifier-Free Guidance (CFG).

        Algorithm (CFG enabled, guidance_scale > 1.0)
        ----------------------------------------------
        At each Euler step, the velocity field is evaluated twice:
          v_cond   = VelocityNet(x, t, c_cond)     ← conditional on instruction
          v_uncond = VelocityNet(x, t, c_uncond)   ← conditional on null instruction ""
          v_guided = v_uncond + w * (v_cond - v_uncond)

        where w = guidance_scale.  w = 1.0 reduces to standard sampling.
        This steers the ODE trajectory toward the language instruction without
        running a separate network.

        CFG is only meaningful when the model was trained with instruction dropout
        (cfg_dropout_prob > 0).  Calling sample() with guidance_scale > 1.0 on a
        model trained without dropout will not improve results.

        Parameters
        ----------
        history        : FloatTensor [B, T_hist, 2]
        instructions   : List[str], length B
        n_steps        : Euler integration steps (20-100 recommended)
        n_samples      : independent samples per scene
        guidance_scale : CFG weight w.
                         None  → use cfg.cfg_guidance_scale from FlowMatchingConfig
                         1.0   → CFG disabled (standard conditional sampling)
                         > 1.0 → CFG enabled; recommended range 1.0 – 3.0
        return_traj    : if True, also return per-step intermediate states

        Returns
        -------
        x_pred : FloatTensor [B, n_samples, T_fut, 2]

        traj_history (only when return_traj=True):
            List[FloatTensor [B, n_samples, T_fut, 2]], length n_steps + 1.
        """
        cfg    = self.cfg
        B      = history.size(0)
        T_fut  = cfg.fut_len
        D      = cfg.traj_dim
        device = history.device
        dt     = 1.0 / n_steps

        # Resolve guidance scale: caller overrides config default if provided
        w = guidance_scale if guidance_scale is not None else cfg.cfg_guidance_scale
        # CFG is active only when w > 1.0 AND the model was trained with dropout
        use_cfg = (w > 1.0) and (cfg.cfg_dropout_prob > 0.0)

        # ── Step 1: encode conditional context ───────────────────────────────
        c_cond = self.condition_encoder(history, instructions)    # [B, hidden_dim]

        # ── Step 2 (CFG only): encode unconditional context ──────────────────
        if use_cfg:
            null_instructions = [""] * B
            c_uncond = self.condition_encoder(history, null_instructions)  # [B, hidden_dim]

        # ── Step 3: replicate condition vectors for multi-sample generation ──
        B_eff = B * n_samples

        def _expand_cond(c_in, n_rep: int, b_eff: int):
            """Expand a condition tensor or (c_hist, c_lang) tuple."""
            if isinstance(c_in, tuple):
                return tuple(
                    ci.unsqueeze(1).expand(-1, n_rep, -1).reshape(b_eff, -1)
                    for ci in c_in
                )
            return c_in.unsqueeze(1).expand(-1, n_rep, -1).reshape(b_eff, -1)

        if n_samples > 1:
            c_cond = _expand_cond(c_cond, n_samples, B_eff)
            if use_cfg:
                c_uncond = _expand_cond(c_uncond, n_samples, B_eff)

        # ── Step 4: sample initial noise x_0 ~ N(0, I) ───────────────────────
        x = torch.randn(B_eff, T_fut, D, device=device)

        history_states: List[torch.Tensor] = []
        if return_traj:
            history_states.append(x.view(B, n_samples, T_fut, D).clone())

        # ── Step 5: Euler integration from t=0 to t=1 ────────────────────────
        for i in range(n_steps):
            t_now = torch.full((B_eff,), i * dt, device=device)   # [B_eff]

            v_cond_step, _ = self.velocity_net(x, t_now, c_cond)  # [B_eff, T_fut, D]

            if use_cfg:
                v_uncond_step, _ = self.velocity_net(x, t_now, c_uncond)
                # CFG formula: steer velocity toward the instruction
                v = v_uncond_step + w * (v_cond_step - v_uncond_step)
            else:
                v = v_cond_step

            x = x + v * dt                                         # Euler step

            if return_traj:
                history_states.append(x.view(B, n_samples, T_fut, D).clone())

        # ── Step 6: reshape to [B, n_samples, T_fut, D] ──────────────────────
        x_pred = x.view(B, n_samples, T_fut, D)

        if return_traj:
            return x_pred, history_states
        return x_pred

    # ── Correlated Euler ODE Sampler (for Scorer) ─────────────────────────────
    @torch.no_grad()
    def sample_correlated(
        self,
        history:        torch.Tensor,          # [B, T_hist, 2]
        instructions:   List[str],             # length B
        n_samples:      int   = 20,
        n_steps:        int   = 30,
        noise_scale:    float = 0.5,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate K *correlated* candidate trajectories using anchored noise.

        Standard sample() draws K fully independent x_0 ~ N(0,I), which makes
        the candidates scatter across the entire trajectory space.  That wide
        scatter makes it harder for the Scorer to learn fine-grained ranking
        signals.

        Anchored noise instead ties all K candidates to a shared base:

            x_base   ~ N(0, I)                          [B, T_fut, D]
            ε_k      ~ N(0, I)                          [B*K, T_fut, D]
            x_0^k    = sqrt(1-α²) * x_base + α * ε_k

        where α = noise_scale.

        α = 0  →  all K trajectories start from the same point (identical)
        α = 1  →  degenerates to fully independent sampling
        α ≈ 0.5 (recommended) → K trajectories form a "variation cluster"
                  around a shared central trajectory; the Scorer learns to
                  discriminate between nearby plausible candidates.

        Parameters
        ----------
        history       : FloatTensor [B, T_hist, 2]
        instructions  : List[str], length B
        n_samples     : K — number of correlated candidates per scene
        n_steps       : Euler integration steps
        noise_scale   : α controlling inter-sample diversity
        guidance_scale: CFG weight (None → uses cfg.cfg_guidance_scale)

        Returns
        -------
        FloatTensor [B, n_samples, T_fut, 2]
        """
        cfg    = self.cfg
        B      = history.size(0)
        T_fut  = cfg.fut_len
        D      = cfg.traj_dim
        device = history.device
        dt     = 1.0 / n_steps

        w = guidance_scale if guidance_scale is not None else cfg.cfg_guidance_scale
        use_cfg = (w > 1.0) and (cfg.cfg_dropout_prob > 0.0)

        # ── Encode conditions (once, shared across all K samples) ─────────────
        c_cond = self.condition_encoder(history, instructions)    # [B, hidden_dim]
        if use_cfg:
            c_uncond = self.condition_encoder(history, [""] * B)  # [B, hidden_dim]

        # Expand condition vectors for K samples: [B, D] → [B*K, D]
        B_eff = B * n_samples

        def _expand_c(c_in, n_rep: int, b_eff: int):
            if isinstance(c_in, tuple):
                return tuple(
                    ci.unsqueeze(1).expand(-1, n_rep, -1).reshape(b_eff, -1)
                    for ci in c_in
                )
            return c_in.unsqueeze(1).expand(-1, n_rep, -1).reshape(b_eff, -1)

        c_cond_eff = _expand_c(c_cond, n_samples, B_eff)
        if use_cfg:
            c_uncond_eff = _expand_c(c_uncond, n_samples, B_eff)

        # ── Anchored noise construction ───────────────────────────────────────
        # x_base is shared across all K samples for the same scene
        x_base = torch.randn(B, T_fut, D, device=device)           # [B, T, D]
        # Expand base to [B*K, T, D]: each scene's base is repeated K times
        x_base_exp = x_base.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(B_eff, T_fut, D)

        # Independent per-sample perturbations
        eps = torch.randn(B_eff, T_fut, D, device=device)

        # Combine: x_0^k = sqrt(1-α²)*x_base + α*ε_k
        alpha = noise_scale
        x = math.sqrt(1.0 - alpha ** 2) * x_base_exp + alpha * eps   # [B*K, T, D]

        # ── Euler integration ─────────────────────────────────────────────────
        for i in range(n_steps):
            t_now = torch.full((B_eff,), i * dt, device=device)
            v_cond, _ = self.velocity_net(x, t_now, c_cond_eff)

            if use_cfg:
                v_uncond, _ = self.velocity_net(x, t_now, c_uncond_eff)
                v = v_uncond + w * (v_cond - v_uncond)
            else:
                v = v_cond

            x = x + v * dt

        return x.view(B, n_samples, T_fut, D)


# ══════════════════════════════════════════════════════════════════════════════
# Flow Matching Inference  (ODE integration)
# ══════════════════════════════════════════════════════════════════════════════
class FlowMatchingInference(nn.Module):
    """
    Wraps a trained FlowMatchingTrainer and integrates the learned ODE
    from x_0 ~ N(0, I) to a predicted trajectory x_1.

    Two solvers are provided:
      "euler" — fixed-step Euler method (fast, good enough for moderate n_steps)
      "rk4"   — 4th-order Runge-Kutta  (more accurate, 4× more NFE per step)

    Parameters
    ----------
    trainer  : trained FlowMatchingTrainer
    n_steps  : number of ODE integration steps (default 50)
    solver   : "euler" or "rk4"
    """

    def __init__(
        self,
        trainer:  FlowMatchingTrainer,
        n_steps:  int = 50,
        solver:   str = "euler",
    ) -> None:
        super().__init__()
        assert solver in ("euler", "rk4"), "solver must be 'euler' or 'rk4'"
        self.trainer  = trainer
        self.n_steps  = n_steps
        self.solver   = solver

    @torch.no_grad()
    def forward(
        self,
        history:      torch.Tensor,   # [B, T_hist, 2]
        instructions: List[str],      # length B
        n_samples:    int = 1,        # samples per scene (for multi-modal eval)
    ) -> torch.Tensor:
        """
        Integrate the ODE and return predicted trajectories.

        Returns
        -------
        x_pred : FloatTensor [B, n_samples, T_fut, 2]
            Predicted future trajectories.  When n_samples=1 the second
            dimension can be squeezed away.
        """
        cfg     = self.trainer.cfg
        B       = history.size(0)
        T_fut   = cfg.fut_len
        D       = cfg.traj_dim
        device  = history.device

        # Encode conditions once (shared across all samples)
        c = self.trainer.condition_encoder(history, instructions)

        # Repeat c for multi-sample generation
        B_eff = B * n_samples
        if n_samples > 1:
            if isinstance(c, tuple):
                c = tuple(
                    ci.unsqueeze(1).expand(-1, n_samples, -1).reshape(B_eff, -1)
                    for ci in c
                )
            else:
                c = c.unsqueeze(1).expand(-1, n_samples, -1).reshape(B_eff, -1)

        # Start from pure noise
        x = torch.randn(B_eff, T_fut, D, device=device)

        # Time grid: integrate from t=0 to t=1
        dt = 1.0 / self.n_steps

        if self.solver == "euler":
            for i in range(self.n_steps):
                t_now = torch.full((B_eff,), i * dt, device=device)
                v, _  = self.trainer.velocity_net(x, t_now, c)
                x     = x + dt * v

        else:  # RK4
            for i in range(self.n_steps):
                t_now = torch.full((B_eff,), i * dt,           device=device)
                t_mid = torch.full((B_eff,), (i + 0.5) * dt,  device=device)
                t_end = torch.full((B_eff,), (i + 1.0) * dt,  device=device)

                k1, _ = self.trainer.velocity_net(x,                  t_now, c)
                k2, _ = self.trainer.velocity_net(x + 0.5 * dt * k1,  t_mid, c)
                k3, _ = self.trainer.velocity_net(x + 0.5 * dt * k2,  t_mid, c)
                k4, _ = self.trainer.velocity_net(x + dt * k3,        t_end, c)

                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Reshape output to [B, n_samples, T_fut, D]
        x_pred = x.view(B, n_samples, T_fut, D)
        return x_pred