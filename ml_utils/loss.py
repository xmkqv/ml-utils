# Loss calculators


import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al. 2017).

    Down-weights easy examples, focuses learning on hard negatives.
    Effective for imbalanced multi-class classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
               gamma=0 equivalent to cross-entropy. Default: 2.0
        alpha: Per-class weights tensor [C]. Optional.
        label_smoothing: Smoothing factor for soft labels. 0.0 = hard labels.
        num_classes: Number of classes (required if label_smoothing > 0).
        reduction: Specifies reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Tensor | None = None,
        label_smoothing: float = 0.0,
        num_classes: int | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.reduction = reduction
        self._alpha: Tensor | None = alpha.float() if alpha is not None else None
        if label_smoothing > 0.0 and num_classes is None:
            msg = "num_classes required when label_smoothing > 0"
            raise ValueError(msg)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss."""
        if self.label_smoothing > 0.0:
            return self._forward_smooth(logits, targets)
        return self._forward_hard(logits, targets)

    def _forward_hard(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Standard focal loss with hard labels."""
        ce_loss = nnf.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self._alpha is not None:
            alpha_t = self._alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss
        return self._reduce(focal_loss)

    def _forward_smooth(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Focal loss with label smoothing."""
        assert self.num_classes is not None
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.label_smoothing / self.num_classes
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = nnf.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        if self._alpha is not None:
            alpha_t = self._alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        focal_loss = focal_weight * ce_loss
        return self._reduce(focal_loss)

    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply reduction."""
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class SDPAttention(nn.Module):
    """Scaled Dot Product Attention using PyTorch's fused SDPA.

    Provides 20-110% memory savings and 10-70% speedup over naive attention.
    Automatically selects the best backend (FlashAttention, Efficient, Math).

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
        backend: Force specific backend ("flash", "efficient", "math", or None for auto)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = (
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(msg)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.backend = backend

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass using fused SDPA."""
        B, L, _ = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dropout_p = self.dropout if self.training else 0.0
        attn_output = self._sdpa(q, k, v, attn_mask, dropout_p, is_causal)

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(attn_output)

    def _sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
        dropout_p: float,
        is_causal: bool,
    ) -> Tensor:
        """Scaled dot-product attention with optional backend selection."""
        if self.backend:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backend_map = {
                "flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH,
            }
            with sdpa_kernel([backend_map[self.backend]]):
                return nnf.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        return nnf.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention for efficient inference.

    Uses single key/value head shared across all query heads.
    Significant memory and compute savings for inference.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of query heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = (
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
            raise ValueError(msg)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass using multi-query attention."""
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, 1, self.head_dim)
        v = self.v_proj(x).reshape(B, L, 1, self.head_dim)

        k = k.expand(B, L, self.num_heads, self.head_dim)
        v = v.expand(B, L, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        attn_output = nnf.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(attn_output)
