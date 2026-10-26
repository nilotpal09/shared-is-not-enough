"""Gated contrastive (SimCLR-style InfoNCE) objective — the objective-class control.

This module is a PURE ADDITION. Nothing in the JEPA loss path imports it, and it
imports nothing that it modifies: the encoder is the standard `vit_nano` built by
`src.helper.init_model`, the data path is the standard calo paired loader, and
the EMA/predictor machinery is simply never constructed. The only new trainable
parameters are the projection head defined here, which is saved under its own
checkpoint key so a probe that loads `ckpt['encoder']` sees exactly the same
state dict shape a JEPA run produces.

What the objective is
---------------------
Two views of the same event (the calo loader's context/target pair) are encoded
by the SAME online encoder, mean-pooled over the token grid, pushed through a
2-layer projection head, L2-normalized, and scored against every other event in
the batch with a temperature-scaled cross-entropy. The paired view is the
positive; all other events (both of their views) are negatives; the sample
itself is excluded. This is SimCLR's NT-Xent, symmetrized over the 2B rows.

Why mean-pool over the token grid
---------------------------------
Because that is exactly the pooling the v2 probe applies at read time
(`conditions/src/_common.py::encode_meanpool_norm` does `enc(x).mean(dim=1)`
then L2-normalize). Training and probing therefore look at the same summary,
and the encoder still emits the standard, unmodified token grid.

Why a 4x-wide GELU MLP for the head
-----------------------------------
No house precedent for a projection head exists in this tree (grep: no
InfoNCE/SimCLR/NT-Xent/temperature anywhere), so the head is matched to the
architecture already in the file: the ViT block's own MLP shape — Linear(D, 4D)
-> GELU -> Linear(4D, D), trunc-normal(0.02) init, biases zero, no BatchNorm
(there is no BatchNorm anywhere in this codebase, and a BN here would make the
loss depend on batch composition in a second, uncontrolled way).

Temperature defaults to 0.2 (the MoCo-v2 / SimCLR-family value); there is no
house precedent to inherit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensors import trunc_normal_


class ProjectionHead(nn.Module):
    """2-layer MLP head. Contrastive-only: saved separately, never part of the
    encoder state dict, and discarded at probe time."""

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, init_std=0.02):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim else 4 * self.in_dim
        self.out_dim = int(out_dim) if out_dim else self.in_dim
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=init_std)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def pool_tokens(h):
    """(B, N, D) standard token grid -> (B, D). Mean over tokens — the pooling
    the v2 probe uses."""
    return h.mean(dim=1)


def info_nce(p1, p2, temperature=0.2):
    """SimCLR NT-Xent over the 2B concatenated views.

    Args
    ----
    p1, p2 : (B, P) projected embeddings of view 1 and view 2, same event order.
    temperature : softmax temperature.

    Returns (loss, stats). Row i of the logits is anchor i; its target is its
    paired view; its own diagonal entry is masked out, so the negatives are
    every other event's two views and nothing else.

    Chance level for reference: loss = ln(2B - 1), accuracy = 1 / (2B - 1).
    """
    assert p1.shape == p2.shape, f'view shapes differ: {p1.shape} vs {p2.shape}'
    assert p1.dim() == 2, f'expected (B, P) embeddings, got {tuple(p1.shape)}'
    B = p1.shape[0]
    dev = p1.device

    z = F.normalize(torch.cat([p1, p2], dim=0).float(), dim=-1)   # (2B, P)
    sim = z @ z.t()                                               # cosine sims
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=dev)
    logits = (sim / float(temperature)).masked_fill(self_mask, float('-inf'))
    targets = torch.cat([torch.arange(B, 2 * B, device=dev),
                         torch.arange(0, B, device=dev)])
    loss = F.cross_entropy(logits, targets)

    with torch.no_grad():
        rows = torch.arange(2 * B, device=dev)
        pos = sim[rows, targets]
        neg_mask = ~self_mask
        neg_mask[rows, targets] = False           # negatives exclude the pair
        stats = {
            'nce_pos_sim': float(pos.mean()),
            'nce_neg_sim': float(sim[neg_mask].mean()),
            'nce_acc': float((logits.argmax(dim=1) == targets).float().mean()),
            'nce_embed_std': float(z.std(dim=0).mean()),
            'nce_chance_loss': float(torch.log(torch.tensor(2.0 * B - 1.0))),
        }
    return loss, stats
