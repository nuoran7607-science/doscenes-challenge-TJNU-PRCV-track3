"""
encoders.py
===========
Conditional encoders for the Flow Matching trajectory prediction model.
(doScenes Challenge — Track 2: Language + History)

Components
----------
1. HistoryEncoder            — BiGRU-based history encoder  [B, T, 2]  → [B, D]
2. LanguageEncoder           — DistilBERT instruction encoder → [B, D]
3. FusionModule              — bidirectional cross-attention fusion → [B, D]
4. ConditionEncoder          — convenience wrapper returning dual-guidance context

Data shapes (consistent with dataset.py)
-----------------------------------------
  history : FloatTensor [B, OBS_LEN=4, 2]
  future  : FloatTensor [B, FUT_LEN=12, 2]   (only used by the flow matching loss)

The cleaned training pipeline always uses BiGRU history encoding, a CLS-pooled
language token for fusion, and dual-guidance context output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast


# ══════════════════════════════════════════════════════════════════════════════
# 1. HistoryEncoder
# ══════════════════════════════════════════════════════════════════════════════
class HistoryEncoder(nn.Module):
    """
    Encodes a past trajectory into a fixed-size context vector.

    Architecture
    ------------
    Linear input projection
      → Bidirectional GRU (n_layers)
      → Project [B, T, 2*hidden_dim] → [B, T, hidden_dim]
      → Learned attention pooling → [B, hidden_dim]

    The encoder returns **both** a pooled global vector and the per-step
    sequence tensor, so the FusionModule can run cross-attention over the
    time axis.

    Parameters
    ----------
    d_in : int
        Dimensionality of each trajectory point (default 2 for x, y).
    hidden_dim : int
        Output feature dimension.
    n_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers (only applied when n_layers > 1).
    """

    def __init__(
        self,
        d_in: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Step 1 — Lift raw coordinates into a richer feature space
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Step 2 — Bidirectional GRU captures both past → future and
        #           future → past temporal dependencies
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Step 3 — Merge the two directions back to hidden_dim
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Step 4 — Soft attention pooling: learn which timestep matters most
        self.attn_pool = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        history : FloatTensor [B, T, d_in]

        Returns
        -------
        pooled   : FloatTensor [B, hidden_dim]    global context vector
        sequence : FloatTensor [B, T, hidden_dim] per-step features for cross-attention
        """
        x = self.input_proj(history)         # [B, T, hidden_dim]
        out, _ = self.gru(x)                 # [B, T, hidden_dim * 2]
        seq = self.out_proj(out)             # [B, T, hidden_dim]

        # Attention-weighted sum over T
        scores = self.attn_pool(seq)         # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * seq).sum(dim=1)  # [B, hidden_dim]

        return pooled, seq


# ══════════════════════════════════════════════════════════════════════════════
# 2. LanguageEncoder
# ══════════════════════════════════════════════════════════════════════════════
class LanguageEncoder(nn.Module):
    """
    Encodes natural-language driving instructions using DistilBERT.

    Architecture
    ------------
    DistilBERT (6 transformer layers, mostly frozen)
      → [CLS] token  (768-d)  → Two-layer MLP projection → [B, hidden_dim]   (pooled)
      → all tokens   (768-d)  → Linear + LayerNorm       → [B, L, hidden_dim] (lang_seq)
      → padding mask                                      → [B, L] bool        (padding_mask)

    Freezing strategy
    -----------------
    DistilBERT has 6 transformer blocks (indices 0-5).
    The first (6 - n_unfreeze_layers) blocks are kept frozen.
    Only the last `n_unfreeze_layers` blocks and the projection heads are
    trained, significantly reducing GPU memory and preventing over-fitting
    on the small doScenes training split.

    Parameters
    ----------
    hidden_dim : int
        Output feature dimension.
    n_unfreeze_layers : int
        Number of DistilBERT transformer blocks to fine-tune from the top.
        Recommended: 1–2 for small datasets.
    max_length : int
        Tokenizer maximum sequence length (truncation applied beyond this).
    """

    _PRETRAINED = "distilbert-base-uncased"
    _N_BERT_LAYERS = 6   # DistilBERT-base has exactly 6 transformer blocks

    def __init__(
        self,
        hidden_dim: int = 128,
        n_unfreeze_layers: int = 2,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        # Load pre-trained weights
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self._PRETRAINED)
        self.bert = DistilBertModel.from_pretrained(self._PRETRAINED)

        # Freeze all parameters first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Selectively unfreeze the last n_unfreeze_layers transformer blocks
        n_freeze = self._N_BERT_LAYERS - n_unfreeze_layers
        for layer in self.bert.transformer.layer[n_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        bert_dim = self.bert.config.hidden_size  # 768

        # CLS-level global projection: 768 → hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Token-level projection: 768 → hidden_dim  (applied to every token position)
        self.token_proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, instructions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        instructions : List[str]   length B
            Raw instruction strings (no pre-tokenization needed).

        Returns
        -------
        pooled       : FloatTensor [B, hidden_dim]
            Global CLS-based summary vector (same as before).
        lang_seq     : FloatTensor [B, L, hidden_dim]
            Per-token projected features for fine-grained cross-attention.
        padding_mask : BoolTensor [B, L]
            True at padding positions (for use as key_padding_mask in MHA).
        """
        device = next(self.bert.parameters()).device
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden     = outputs.last_hidden_state          # [B, L, 768]

        cls_token      = all_hidden[:, 0, :]                # [B, 768]
        pooled         = self.proj(cls_token)               # [B, hidden_dim]

        lang_seq       = self.token_proj(all_hidden)        # [B, L, hidden_dim]

        # padding_mask: True = ignore this position (PyTorch MHA convention)
        padding_mask   = attention_mask == 0                # [B, L], bool

        return pooled, lang_seq, padding_mask


# ══════════════════════════════════════════════════════════════════════════════
# 3. FusionModule
# ══════════════════════════════════════════════════════════════════════════════
class FusionModule(nn.Module):
    """
    Fuses history and language conditions via bidirectional cross-attention.

    Two cross-attention passes (token-level, upgraded from single-vector)
    ---------------------------------------------------------------------
    Pass 1 — Language-tokens-attend-to-History:
        Query  = lang_seq             [B, L, D]   all language tokens
        Key/V  = hist_seq             [B, T, D]
        Output = residual-updated lang tokens [B, L, D]
               → attention-pooled (with padding mask) → lang_ctx [B, D]

    Pass 2 — History-attends-to-Language-tokens:
        Query  = hist_pooled          [B, 1, D]
        Key/V  = lang_seq             [B, L, D]   all language tokens
        key_padding_mask              [B, L]       ignore padding positions
        Output → hist_ctx [B, D]

    Both outputs are merged through a two-layer MLP to produce the final
    condition vector for the velocity field network (when return_dual=False).
    When return_dual=True the two enriched vectors are returned separately for
    dual-guidance decoding.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension (must match both encoder outputs).
    n_heads : int
        Number of multi-head attention heads.  hidden_dim must be divisible by n_heads.
    dropout : float
        Dropout applied inside attention and the fusion MLP.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        attn_kwargs = dict(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention 1: all language token queries → history sequence
        self.lang_attends_hist = nn.MultiheadAttention(**attn_kwargs)

        # Cross-attention 2: history query → all language token sequence
        self.hist_attends_lang = nn.MultiheadAttention(**attn_kwargs)

        # Post-attention layer norms
        self.norm_lang = nn.LayerNorm(hidden_dim)
        self.norm_hist = nn.LayerNorm(hidden_dim)

        # Attention pooling: compress updated lang tokens [B, L, D] → [B, D]
        self.lang_seq_pool = nn.Linear(hidden_dim, 1)

        # Final fusion MLP: [2*D] → [D]  (only used when return_dual=False)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        lang_feat:         torch.Tensor,
        hist_pooled:       torch.Tensor,
        hist_seq:          torch.Tensor,
        lang_seq:          torch.Tensor,
        lang_padding_mask: Optional[torch.Tensor] = None,
        return_dual:       bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        lang_feat         : kept for backward-compat; not used in forward pass.
        hist_pooled       : global history vector from HistoryEncoder  [B, D].
        hist_seq          : per-step history features  [B, T_hist, D].
        lang_seq          : per-token language features  [B, L, D].
        lang_padding_mask : True at padding positions (from tokenizer attention_mask==0).
        return_dual       : when True, return (hist_ctx, lang_ctx) separately instead
                            of the fused context vector.  Both vectors have been
                            mutually enriched by cross-attention.

        Returns
        -------
        context                        : FloatTensor [B, D]   (return_dual=False)
        (hist_ctx, lang_ctx)           : Tuple[Tensor, Tensor] [B, D] each  (return_dual=True)
        """
        # ── Pass 1: all language tokens query history timesteps ──────────────
        lang_attn_out, _ = self.lang_attends_hist(
            query=lang_seq, key=hist_seq, value=hist_seq
        )                                                    # [B, L, D]
        lang_seq_updated = self.norm_lang(lang_seq + lang_attn_out)  # [B, L, D]

        # Attention-weighted pool [B, L, D] → [B, D], respecting padding
        pool_scores = self.lang_seq_pool(lang_seq_updated)   # [B, L, 1]
        if lang_padding_mask is not None:
            pool_scores = pool_scores.masked_fill(
                lang_padding_mask.unsqueeze(-1), float("-inf")
            )
        pool_weights = torch.softmax(pool_scores, dim=1)     # [B, L, 1]
        lang_ctx = (pool_weights * lang_seq_updated).sum(dim=1)  # [B, D]

        # ── Pass 2: history query attends over all language tokens ───────────
        hist_q = hist_pooled.unsqueeze(1)                    # [B, 1, D]
        hist_attn_out, _ = self.hist_attends_lang(
            query=hist_q,
            key=lang_seq,
            value=lang_seq,
            key_padding_mask=lang_padding_mask,
        )                                                    # [B, 1, D]
        hist_ctx = self.norm_hist(
            hist_pooled + hist_attn_out.squeeze(1)
        )                                                    # [B, D]

        if return_dual:
            # Return the two cross-attention-enriched vectors separately.
            # Bypasses fusion_mlp so each signal stays independent for the
            # dual-guidance VelocityNetwork.
            return hist_ctx, lang_ctx

        # ── Merge both enriched representations ──────────────────────────────
        combined = torch.cat([lang_ctx, hist_ctx], dim=-1)   # [B, 2*D]
        context  = self.fusion_mlp(combined)                 # [B, D]

        return context


# ══════════════════════════════════════════════════════════════════════════════
# 4. ConditionEncoder (unified wrapper)
# ══════════════════════════════════════════════════════════════════════════════
class ConditionEncoder(nn.Module):
    """
    Convenience wrapper that bundles history encoder + language encoder + fusion.

    Usage
    -----
    encoder = ConditionEncoder(hidden_dim=128)
    c_hist, c_lang = encoder(history, instructions)   # → ([B, 128], [B, 128])

    The cleaned pipeline always returns a dual-guidance tuple. The fusion MLP is
    bypassed so history and language signals remain independent for per-layer
    injection in the velocity network.

    Parameters
    ----------
    hidden_dim               : shared feature dimension across all sub-modules
    d_in                     : trajectory input dimensionality (2 for x, y)
    n_hist_layers            : depth of HistoryEncoder GRU layers
    n_unfreeze_bert          : DistilBERT blocks to fine-tune from top
    n_heads                  : attention heads for FusionModule
    dropout                  : global dropout rate
    max_length               : tokenizer max sequence length for LanguageEncoder
    obs_len                  : accepted for call-site compatibility; BiGRU does
                               not need a fixed history length.
    """

    def __init__(
        self,
        hidden_dim:              int   = 128,
        d_in:                    int   = 2,
        n_hist_layers:           int   = 2,
        n_unfreeze_bert:         int   = 2,
        n_heads:                 int   = 4,
        dropout:                 float = 0.1,
        max_length:              int   = 64,
        obs_len:                 int   = 4,
    ) -> None:
        super().__init__()
        _ = obs_len  # Kept so older call sites do not need a signature change.

        self.history_encoder = HistoryEncoder(
            d_in=d_in,
            hidden_dim=hidden_dim,
            n_layers=n_hist_layers,
            dropout=dropout,
        )

        self.language_encoder = LanguageEncoder(
            hidden_dim=hidden_dim,
            n_unfreeze_layers=n_unfreeze_bert,
            max_length=max_length,
        )
        self.fusion = FusionModule(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self,
        history: torch.Tensor,    # [B, T_hist, 2]
        instructions: List[str],  # length B
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        history      : FloatTensor [B, T_hist, 2]
        instructions : List[str] of length B

        Returns
        -------
        (c_hist, c_lang)     : Tuple[FloatTensor, FloatTensor]
            c_hist and c_lang are each [B, hidden_dim]; both have been
            cross-attention-enriched with the other modality before being
            returned separately for dual-guidance per-layer injection.
        """
        hist_pooled, hist_seq            = self.history_encoder(history)
        lang_pooled, lang_seq, lang_mask = self.language_encoder(instructions)

        # The cleaned pipeline uses the CLS-pooled language vector for fusion.
        lang_seq_in  = lang_pooled.unsqueeze(1)
        lang_mask_in = None

        result = self.fusion(
            lang_feat=lang_pooled,
            hist_pooled=hist_pooled,
            hist_seq=hist_seq,
            lang_seq=lang_seq_in,
            lang_padding_mask=lang_mask_in,
            return_dual=True,
        )
        return result

    def trainable_parameters(self):
        """Yields only the parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)