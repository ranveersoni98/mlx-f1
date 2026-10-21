"""MLX race model: score every driver, softmax across the field."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .features import PL_K


def masked_log_softmax(scores: mx.array, mask: mx.array) -> mx.array:
    neg_inf = mx.array(-1e9, dtype=scores.dtype)
    scores = mx.where(mask, scores, neg_inf)
    return scores - mx.logsumexp(scores, axis=-1, keepdims=True)


class RaceNet(nn.Module):
    """A small MLP over driver features plus a learned circuit embedding."""

    def __init__(
        self,
        n_features: int,
        n_circuits: int,
        hidden: int = 64,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.out = nn.Linear(hidden // 2, 1)
        self.circuit_emb = nn.Embedding(n_circuits, embed_dim)
        self.emb_proj = nn.Linear(embed_dim, hidden)

    def __call__(self, x: mx.array, circuit_idx: mx.array) -> mx.array:
        hidden = nn.relu(self.fc1(x))
        circuit = self.emb_proj(self.circuit_emb(circuit_idx))[:, None, :]
        hidden = nn.relu(self.fc2(hidden + circuit))
        return self.out(hidden).squeeze(-1)


def loss_terms(
    model: RaceNet,
    x: mx.array,
    mask: mx.array,
    winner: mx.array,
    circuit_idx: mx.array,
    rank_idx: mx.array,
    rank_valid: mx.array,
    *,
    pl_weight: float = 0.3,
    pl_k: int = PL_K,
) -> tuple[mx.array, mx.array, mx.array]:
    """Winner cross-entropy plus a truncated Plackett-Luce ranking loss."""
    scores = model(x, circuit_idx)
    log_probs = masked_log_softmax(scores, mask)
    winner_log_prob = mx.take_along_axis(
        log_probs, winner[:, None], axis=1
    ).squeeze(1)
    win_loss = -mx.mean(winner_log_prob)

    selected = mx.zeros_like(mask)
    pl_loss = mx.array(0.0)
    field = mx.arange(scores.shape[1])
    for k in range(pl_k):
        remaining = mx.logical_and(mask, mx.logical_not(selected))
        step_log_probs = masked_log_softmax(scores, remaining)
        target = rank_idx[:, k]
        target_log_prob = mx.take_along_axis(
            step_log_probs, target[:, None], axis=1
        ).squeeze(1)
        valid = rank_valid[:, k].astype(scores.dtype)
        pl_loss = pl_loss - mx.sum(target_log_prob * valid) / mx.maximum(
            mx.sum(valid), 1.0
        )
        chosen = mx.logical_and(
            field[None, :] == target[:, None],
            rank_valid[:, k][:, None],
        )
        selected = mx.logical_or(selected, chosen)
    pl_loss = pl_loss / pl_k

    total = win_loss + pl_weight * pl_loss
    return total, win_loss, pl_loss


def total_loss(
    model: RaceNet,
    x: mx.array,
    mask: mx.array,
    winner: mx.array,
    circuit_idx: mx.array,
    rank_idx: mx.array,
    rank_valid: mx.array,
) -> mx.array:
    return loss_terms(
        model, x, mask, winner, circuit_idx, rank_idx, rank_valid
    )[0]


def probabilities(
    model: RaceNet, x: mx.array, mask: mx.array, circuit_idx: mx.array
) -> mx.array:
    scores = model(x, circuit_idx)
    scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
    return mx.softmax(scores, axis=-1)
