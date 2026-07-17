"""
LitCVit Architecture – PyTorch implementation based on:
  Nisa et al., "LitCVit: A Lightweight Self-Supervised Contrastive Vision
  Transformer for Encrypted Malicious Traffic Detection",
  IEEE Trans. Inf. Forensics Security, Vol. 21, 2026.

Components (Section IV):
  1. Convolutional Patchify Stem  (depthwise-separable convolutions)
  2. Hierarchical wf-Attention modules
       – Byte-level  : windowed factorized attention
       – Packet-level: inter-packet windowed factorized attention
       – Flow-level  : global factorized attention + CLS token
  3. Contrastive Embedding Head  (MLP projection, used during pre-training)
  4. Classification Head         (3-layer MLP, used during fine-tuning)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Convolutional Patchify Stem
# ─────────────────────────────────────────────────────────────────────────────

class ConvPatchifyStem(nn.Module):
    """
    Depthwise-separable convolution tokeniser (Algorithm 1, lines 1-3).
    Input:  (B, 1, H, W)  – grayscale ET-flow image
    Output: (B, N, d)     – patch token sequence
    """

    def __init__(self, in_ch: int = 1, emb_dim: int = 192, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size

        # depthwise conv  (spatial filtering, one filter per input channel)
        self.dw_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=patch_size, stride=patch_size,
            padding=0, groups=in_ch, bias=False
        )
        # pointwise conv  (channel mixing / projection to emb_dim)
        self.pw_conv = nn.Conv2d(in_ch, emb_dim, kernel_size=1, bias=True)
        self.norm    = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        x = self.dw_conv(x)            # (B, 1, H/p, W/p)
        x = self.pw_conv(x)            # (B, d, H/p, W/p)
        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, d),  N = h*w
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2a.  Low-Rank Factorized Self-Attention  (Section IV-B-2)
# ─────────────────────────────────────────────────────────────────────────────

class LowRankFactorizedAttention(nn.Module):
    """
    Attention with low-rank projections Q,K,V ∈ R^{d×r}  (r << d).
    Global version – used at the flow level (CLS-aware).
    """

    def __init__(self, dim: int, rank: int = 16, attn_drop: float = 0.0):
        super().__init__()
        self.rank  = rank
        self.scale = rank ** -0.5
        self.tau   = nn.Parameter(torch.ones(1) * math.log(1 / 0.07))

        self.wq = nn.Linear(dim, rank, bias=False)
        self.wk = nn.Linear(dim, rank, bias=False)
        self.wv = nn.Linear(dim, rank, bias=False)
        self.proj = nn.Linear(rank, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.wq(x)          # (B, N, r)
        K = self.wk(x)
        V = self.wv(x)
        attn = (Q @ K.transpose(-2, -1)) * (self.tau.exp() * self.scale)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out  = attn @ V         # (B, N, r)
        return self.proj(out)   # (B, N, d)


# ─────────────────────────────────────────────────────────────────────────────
# 2b.  Windowed Factorized Attention  (Section IV-B-2)
#      Used at byte and packet levels to restrict attention to local windows.
# ─────────────────────────────────────────────────────────────────────────────

class WindowedFactorizedAttention(nn.Module):
    """
    Restricts low-rank factorized attention to windows of size `window`.
    Complexity: O(N · w)  vs  O(N²) for full attention.
    """

    def __init__(self, dim: int, rank: int = 16, window: int = 8,
                 attn_drop: float = 0.0):
        super().__init__()
        self.window = window
        self.rank   = rank
        self.scale  = rank ** -0.5
        self.tau    = nn.Parameter(torch.ones(1) * math.log(1 / 0.07))

        self.wq   = nn.Linear(dim, rank, bias=False)
        self.wk   = nn.Linear(dim, rank, bias=False)
        self.wv   = nn.Linear(dim, rank, bias=False)
        self.proj = nn.Linear(rank, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        w = self.window

        # Pad sequence so it's divisible by window size
        pad = (w - N % w) % w
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        _, N_pad, _ = x.shape
        n_wins = N_pad // w

        # Reshape into windows: (B * n_wins, w, d)
        x_win = x.view(B, n_wins, w, d).reshape(B * n_wins, w, d)

        Q = self.wq(x_win)          # (B*nw, w, r)
        K = self.wk(x_win)
        V = self.wv(x_win)
        attn = (Q @ K.transpose(-2, -1)) * (self.tau.exp() * self.scale)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out  = attn @ V              # (B*nw, w, r)
        out  = self.proj(out)        # (B*nw, w, d)
        out  = out.view(B, N_pad, d)

        # Remove padding
        if pad > 0:
            out = out[:, :N, :]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2c.  Transformer Block  (LN + Attention + LN + FFN)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, attn: nn.Module, mlp_ratio: float = 2.0,
                 drop: float = 0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = attn
        self.norm2  = nn.LayerNorm(dim)
        hidden_dim  = int(dim * mlp_ratio)
        self.ffn    = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Hierarchical wf-Attention Encoder  (Algorithm 1, lines 5-14)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalWFAttentionEncoder(nn.Module):
    """
    Three-stage hierarchical encoder:
      Stage 1 – Byte-level   : local windowed factorized attention (window = byte_win)
      Stage 2 – Packet-level : inter-packet windowed attention (window = pkt_win)
      Stage 3 – Flow-level   : global factorized attention across all tokens + CLS
    Between stages: depthwise conv to inject spatial bias + token compression.
    """

    def __init__(self, dim: int = 192, rank: int = 16,
                 byte_win: int = 8, pkt_win: int = 5, drop: float = 0.0):
        super().__init__()

        # ── Stage 1: byte-level windowed attention ───────────────────────────
        byte_attn   = WindowedFactorizedAttention(dim, rank, byte_win)
        self.byte_enc = TransformerBlock(dim, byte_attn, drop=drop)

        # Transition conv: spatial inductive bias, compresses byte → packet
        self.byte2pkt_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.LayerNorm(dim),    # applied after transpose
        )

        # ── Stage 2: packet-level windowed attention ─────────────────────────
        pkt_attn    = WindowedFactorizedAttention(dim, rank, pkt_win)
        self.pkt_enc  = TransformerBlock(dim, pkt_attn, drop=drop)

        # Transition conv: packet → flow level
        self.pkt2flow_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

        # ── Stage 3: flow-level global factorized attention ──────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        flow_attn       = LowRankFactorizedAttention(dim, rank)
        self.flow_enc   = TransformerBlock(dim, flow_attn, drop=drop)

        self.norm = nn.LayerNorm(dim)

    def _apply_conv1d_ln(self, conv_seq, x):
        """Helper: apply Conv1d ops (need (B,C,N) layout) then LayerNorm."""
        x = x.transpose(1, 2)          # (B, d, N)
        for layer in conv_seq:
            if isinstance(layer, nn.LayerNorm):
                x = x.transpose(1, 2)   # back to (B, N, d)
                x = layer(x)
                x = x.transpose(1, 2)   # (B, d, N)
            else:
                x = layer(x)
        return x.transpose(1, 2)        # (B, N', d)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, d) patch tokens from stem.
        Returns:
            cls_out  : (B, d)  – CLS token for contrastive / classification
            all_feats: (B, N3, d) – all flow-level tokens (for downstream use)
        """
        # Stage 1 – byte-level
        x = self.byte_enc(x)
        x = self._apply_conv1d_ln(self.byte2pkt_conv, x)

        # Stage 2 – packet-level
        x = self.pkt_enc(x)
        x_pkt_out = x                          # save for fine-tuning unfreeze

        x_t = x.transpose(1, 2)               # (B, d, N)
        for layer in self.pkt2flow_conv:
            x_t = layer(x_t)
        x = x_t.transpose(1, 2)               # (B, N', d)

        # Stage 3 – flow-level (CLS prepended)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)       # (B, N'+1, d)
        x   = self.flow_enc(x)
        x   = self.norm(x)

        cls_out   = x[:, 0]                    # (B, d)
        all_feats = x[:, 1:]                   # (B, N', d)

        # Store intermediate packet features for partial unfreeze
        self._pkt_out = x_pkt_out

        return cls_out, all_feats


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Contrastive Projection Head  (MLP, used only during pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP projection head for ntXent contrastive loss."""

    def __init__(self, dim: int = 192, proj_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Classification Head  (used during fine-tuning, Algorithm 2)
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    3-layer FC MLP with GELU activations and dropout=0.2.
    Ends in softmax (applied externally via cross-entropy loss).
    """

    def __init__(self, dim: int = 192, n_classes: int = 20,
                 dropout: float = 0.2):
        super().__init__()
        mid = dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Complete LitCVit Model
# ─────────────────────────────────────────────────────────────────────────────

class LitCVit(nn.Module):
    """
    Full LitCVit model.

    Two operational modes:
      mode='pretrain'  – encoder + projection head (ntXent)
      mode='finetune'  – encoder + classification head

    Fine-tuning protocol (Section IV-C):
      Phase 1 (epochs 0-99)  : encoder frozen, only head trained
      Phase 2 (epochs 100+)  : last two encoder blocks (pkt_enc + flow_enc) unfrozen
    """

    def __init__(self,
                 img_h: int = 40, img_w: int = 40,
                 in_ch: int = 1,
                 patch_size: int = 4,
                 emb_dim: int = 192,
                 rank: int = 16,
                 byte_win: int = 8,
                 pkt_win: int = 5,
                 proj_dim: int = 128,
                 n_classes: int = 20,
                 dropout: float = 0.2,
                 mode: str = "pretrain"):
        super().__init__()
        assert mode in ("pretrain", "finetune")
        self.mode    = mode
        self.emb_dim = emb_dim

        # ── Positional encoding ───────────────────────────────────────────────
        n_patches = (img_h // patch_size) * (img_w // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Core modules ──────────────────────────────────────────────────────
        self.stem    = ConvPatchifyStem(in_ch, emb_dim, patch_size)
        self.encoder = HierarchicalWFAttentionEncoder(
            dim=emb_dim, rank=rank,
            byte_win=byte_win, pkt_win=pkt_win, drop=dropout
        )

        if mode == "pretrain":
            self.head = ProjectionHead(emb_dim, proj_dim)
        else:
            self.head = ClassificationHead(emb_dim, n_classes, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        """Freeze the entire encoder (Phase 1 of fine-tuning)."""
        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_two_blocks(self):
        """Unfreeze the last two encoder blocks (Phase 2, epoch ≥ 100)."""
        for p in self.encoder.pkt_enc.parameters():
            p.requires_grad = True
        for p in self.encoder.flow_enc.parameters():
            p.requires_grad = True
        for p in self.encoder.norm.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        Returns logits / projections depending on mode.
        """
        tokens = self.stem(x)
        # Add positional encoding (trim or pad if needed)
        n = min(tokens.size(1), self.pos_embed.size(1))
        tokens[:, :n] = tokens[:, :n] + self.pos_embed[:, :n]

        cls_out, _ = self.encoder(tokens)
        return self.head(cls_out)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw CLS embedding (d-dim) for UMAP / analysis."""
        tokens = self.stem(x)
        n = min(tokens.size(1), self.pos_embed.size(1))
        tokens[:, :n] = tokens[:, :n] + self.pos_embed[:, :n]
        cls_out, _ = self.encoder(tokens)
        return cls_out

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def from_pretrained(ckpt_path: str, n_classes: int,
                        emb_dim: int = 192, rank: int = 16,
                        **kwargs) -> "LitCVit":
        """
        Load encoder weights from a pre-training checkpoint and
        return a fine-tuning model with a fresh classification head.
        """
        model = LitCVit(emb_dim=emb_dim, rank=rank,
                        n_classes=n_classes, mode="finetune", **kwargs)
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        # Load only encoder + stem weights; skip projection head
        new_state = {k: v for k, v in state.items()
                     if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[LitCVit] Loaded pretrained weights from '{ckpt_path}'")
        if missing:
            print(f"  Missing keys : {missing[:5]}{'...' if len(missing)>5 else ''}")
        return model
