"""Gated BYOL objective — the second objective-class control (the ingredient split).

This module is a PURE ADDITION, built to the same rules as `src/contrastive.py`.
Nothing in the JEPA loss path imports it; the encoder is the standard `vit_nano`
built by `src.helper.init_model`; the data path is the standard calo paired
loader. The only new trainable parameters are the projection head and the
prediction head defined here, both saved under their own checkpoint keys so a
probe that loads `ckpt['encoder']` or `ckpt['target_encoder']` sees exactly the
state dict shape a JEPA run produces.

Why this arm exists
-------------------
The study has one JEPA family (masked latent prediction against an EMA target)
and one contrastive family (InfoNCE, no EMA, no predictor, explicit negatives).
JEPA carries three ingredients at once. BYOL keeps two of them and drops the
third, so the three arms together separate which one the channel effect needs:

    ingredient          JEPA   contrastive   BYOL
    -----------------------------------------------
    EMA/self-referential target    yes    no            YES
    prediction head (asymmetry)    yes    no            YES
    explicit negatives             no     yes           no
    masking / latent locality      yes    no            NO

BYOL is therefore JEPA-minus-masking: same self-referential moving target, same
predictor asymmetry, but the summary it predicts is the whole-view pooled
embedding rather than the latents of masked blocks.

What the objective is
---------------------
Two views of the same event (the calo loader's context/target pair). The ONLINE
network is encoder -> mean-pool -> projector -> predictor. The TARGET network is
an EMA copy of encoder -> mean-pool -> projector, with NO predictor, run under
`no_grad`. The loss is the normalized L2 regression BYOL uses,

    D(p, t) = || p/||p|| - t/||t|| ||^2 = 2 - 2 cos(p, t),

taken from each view's online prediction to the OTHER view's target projection
and symmetrized. No negatives are involved at any point: the loss of a batch
element depends only on that element.

Reference lines for reading the numbers, both derivable rather than chosen:

    loss ~= 2.0   prediction and target are uncorrelated (cos ~ 0) — the value at
                  initialization, and the value a run that learns nothing keeps.
    loss -> 0     cos -> 1. This happens BOTH when the objective succeeds AND in
                  total collapse (every event mapped to the same point). ⚠ For
                  this objective a falling loss is NOT by itself evidence of
                  health — that is what `byol_embed_std` and `byol_offdiag_cos`
                  below are for, and why the ep50 gate reads them, not the loss
                  alone.

Collapse instrumentation (the gate reads these)
-----------------------------------------------
`byol_embed_std` is the same statistic `nce_embed_std` is on the contrastive
arm — mean over dimensions of the per-dimension standard deviation, across the
batch, of the L2-normalized embeddings the loss actually sees on the online side
(here the predictions) — so the two twins' collapse numbers sit on the same
scale. A standard deviation subtracts the batch mean, so this statistic is
already mean-centred: it measures how much of the embedding varies FROM EVENT TO
EVENT, which is the thing collapse destroys. `byol_target_std` is the same
statistic on the EMA target projections, i.e. on the other side of the loss.
`byol_spread_frac` is the interpretable normalized version of the same reading:
the fraction of the target embeddings' total norm carried by event-to-event
variation rather than by the direction all events share, ||z - z_bar||_F /
||z||_F. It is 0 in total collapse and 1 if the embeddings had no common
component at all.

⚠ **Two reference lines that do NOT apply on this data, measured rather than
assumed** (smoke, 2026-08-11, and consistent with `src/train.py`'s own comment
that event-pooled calo features "always pick up the same dominant direction"):

  * The uniform-on-sphere line 1/sqrt(P) = 0.1021 is **unreachable here**. At
    initialization, before a single gradient step, `byol_embed_std` on real calo
    batches is ~8e-4 — about 100x below it. A pooled calo embedding is dominated
    by one common direction from the start, so the sphere reference describes no
    state this encoder is ever in, and reading against it would call every run
    collapsed.
  * `byol_offdiag_cos` — mean cosine between DIFFERENT events' target projections
    — reads **0.9999 at initialization**. It is saturated before training begins
    and therefore **cannot be the gate**. It is kept as a logged column for
    continuity only.

The reference that does apply is the run's OWN value at initialization, and the
direction it moves: event-to-event spread that CONTRACTS below its starting value
is the collapse signature; spread that grows is the objective doing work. The
contrastive twin on identical data is the calibration for what growth looks like
(it reached `nce_embed_std` 0.0937 from the same ~1e-3 start).

Head shapes, and one deviation recorded as a deviation
------------------------------------------------------
Both heads are the ViT block's own MLP shape — Linear(D, 4D) -> GELU ->
Linear(4D, D), trunc-normal(0.02), zero biases — identical to the contrastive
arm's projection head, so the only difference between the two twins is the
objective and not the head.

⚠ **Named deviation from the published BYOL recipe.** BYOL as published puts
BatchNorm in both heads, and the role of batch statistics in preventing its
collapse has its own literature. There is no BatchNorm anywhere in this codebase,
and adding one here would (a) break the match with the contrastive twin's head
and (b) make the loss depend on batch composition in a second, uncontrolled way —
the same reason `src/contrastive.py` gives for rejecting it. So this is BYOL's
*mechanism* (EMA target + predictor asymmetry + no negatives) on the house
architecture, NOT a reproduction of the published recipe, and it is more
collapse-prone than the published one. Any reading of this arm is a matched-arm
contrast between byol-indirect and byol-absent under identical conditions; it is
never a claim about what "BYOL does".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tensors import trunc_normal_


class ProjectionHead(nn.Module):
    """2-layer MLP head, online and (as an EMA copy) target side. Saved under its
    own checkpoint key, never part of the encoder state dict, discarded at probe
    time. Byte-for-byte the same shape as the contrastive arm's head."""

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


class PredictionHead(nn.Module):
    """BYOL's predictor q. ONLINE SIDE ONLY — this is the asymmetry that makes
    the objective non-trivial, and the target network must never contain one.
    Same MLP shape as the projection head."""

    def __init__(self, dim, hidden_dim=None, init_std=0.02):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim else 4 * self.dim
        self.fc1 = nn.Linear(self.dim, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=init_std)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@torch.no_grad()
def momentum_update(online, target, m):
    """EMA the target network toward the online one.

    The arithmetic is `src/train.py`'s Step 3 line, unchanged:

        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

    `.detach()` on the online side and `no_grad` on the whole update are both
    deliberate: no gradient ever crosses into the target network.
    """
    for param_q, param_k in zip(online.parameters(), target.parameters()):
        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)


def momentum_schedule(ema, ipe, num_epochs, ipe_scale):
    """The house EMA schedule generator, verbatim from `src/train.py`:

        m = ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)

    linear from `ema[0]` to `ema[1]` over the run. With the standard calo recipe's
    `optimization.ema: [0.996, 1.0]` this starts at 0.996 and ends at 1.0, which
    is also BYOL's published tau_base -> 1 schedule, so nothing is invented.
    """
    return (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
            for i in range(int(ipe*num_epochs*ipe_scale)+1))


def pool_tokens(h):
    """(B, N, D) standard token grid -> (B, D). Mean over tokens — the pooling the
    v2 probe uses (`conditions/src/_common.py::encode_meanpool_norm`), and the
    same pooling the contrastive twin uses. No mask is applied anywhere: the whole
    view is pooled. That absence is the point of this arm."""
    return h.mean(dim=1)


def regression_loss(p, t):
    """BYOL's normalized-L2 regression, 2 - 2 cos(p, t), averaged over the batch.

    `t` is detached here as well as being produced under `no_grad` in the training
    loop: the stop-gradient is the load-bearing part of this objective and is
    enforced in both places on purpose.
    """
    p = F.normalize(p.float(), dim=-1)
    t = F.normalize(t.float().detach(), dim=-1)
    return (2.0 - 2.0 * (p * t).sum(dim=-1)).mean()


def byol_loss(p1, p2, t1, t2):
    """Symmetrized BYOL loss and the statistics the health gate reads.

    Args
    ----
    p1, p2 : (B, P) ONLINE predictions for view 1 and view 2, same event order.
    t1, t2 : (B, P) EMA TARGET projections for view 1 and view 2, same order.

    The cross terms are the whole objective: view 1's prediction regresses view
    2's target and vice versa. Returns (loss, stats).
    """
    assert p1.shape == p2.shape == t1.shape == t2.shape, (
        f'shape mismatch: p1 {tuple(p1.shape)} p2 {tuple(p2.shape)} '
        f't1 {tuple(t1.shape)} t2 {tuple(t2.shape)}')
    assert p1.dim() == 2, f'expected (B, P) embeddings, got {tuple(p1.shape)}'

    loss = 0.5 * (regression_loss(p1, t2) + regression_loss(p2, t1))

    with torch.no_grad():
        B = p1.shape[0]
        zp = F.normalize(torch.cat([p1, p2], dim=0).float(), dim=-1)   # predictions
        zt = F.normalize(torch.cat([t1, t2], dim=0).float(), dim=-1)   # targets
        # cosine each prediction achieves against the target it regresses
        cos = torch.cat([(F.normalize(p1.float(), dim=-1) * F.normalize(t2.float(), dim=-1)).sum(-1),
                         (F.normalize(p2.float(), dim=-1) * F.normalize(t1.float(), dim=-1)).sum(-1)])
        # mean cosine between DIFFERENT events' target projections: ~0 when the
        # representation is spread, -> 1 in collapse. The loss cannot see this.
        sim_t = zt @ zt.t()
        off = ~torch.eye(2 * B, dtype=torch.bool, device=zt.device)
        # exclude the same event's other view too: those are supposed to be close
        rows = torch.arange(2 * B, device=zt.device)
        pair = torch.cat([torch.arange(B, 2 * B, device=zt.device),
                          torch.arange(0, B, device=zt.device)])
        off[rows, pair] = False
        # fraction of the target embeddings' norm carried by event-to-event
        # variation rather than by the direction every event shares
        resid = zt - zt.mean(dim=0, keepdim=True)
        denom = float(zt.norm())
        stats = {
            'byol_cos': float(cos.mean()),
            'byol_embed_std': float(zp.std(dim=0).mean()),      # online predictions
            'byol_target_std': float(zt.std(dim=0).mean()),     # EMA targets
            'byol_spread_frac': float(resid.norm()) / denom if denom > 0 else 0.0,
            'byol_offdiag_cos': float(sim_t[off].mean()),       # saturates: NOT the gate
            'byol_uncorrelated_loss': 2.0,   # the value at cos = 0, for the log line
        }
    return loss, stats
