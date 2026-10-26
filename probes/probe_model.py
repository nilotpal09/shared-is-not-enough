"""The truth-query probe.

One query per TRUE particle, built from that particle's known position (and,
in the ablation, its energy). The queries cross-attend over the frozen
encoder's UNPOOLED token grid and each query emits (ln alpha_hat, ln T_hat)
for its own particle. There is deliberately NO query-to-query attention:
queries are independent given the tokens, which is what makes them
truth-anchored (permuting the queries permutes the predictions exactly) and
what makes the permuted-truth floor meaningful.

Capacity: 'big' = 2 cross-attention blocks, d_model 64 (the paper's main probe
is this trunk with a 3-output head, common.BigProbe).
"""
import math

import torch
import torch.nn as nn

CAPACITY = {
    'small': dict(layers=1, d_model=32, heads=4),
    'big':   dict(layers=2, d_model=64, heads=4),
}
# Positional Fourier octaves in the query. The token grid is 16x16, so octave k
# has period (image width)/2^k; k=4 already resolves one token. Octaves far above
# that add nothing the probe can use to FIND a token and only give it a unique
# fingerprint per particle to memorize.
N_FOURIER = 5


def query_features(x, y, lnE, half_w, half_h, lnE_mu, lnE_sd, use_energy,
                   n_fourier=None):
    """(B, P, F) query features from truth. Position always; energy only in the
    `pose` ablation."""
    n_fourier = N_FOURIER if n_fourier is None else n_fourier
    u = (x / half_w).clamp(-2, 2)
    v = (y / half_h).clamp(-2, 2)
    feats = [u, v]
    for k in range(n_fourier):
        f = (2.0 ** k) * math.pi
        feats += [torch.sin(f * u), torch.cos(f * u),
                  torch.sin(f * v), torch.cos(f * v)]
    if use_energy:
        feats.append((lnE - lnE_mu) / lnE_sd)
    return torch.stack(feats, dim=-1)


class CrossBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.nm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        # learnable strength of the "look near your own particle" prior; the probe
        # can switch it off by driving this to 0
        self.loc = nn.Parameter(torch.tensor(1.0))

    def forward(self, q, kv, bias=None):
        mask = None
        if bias is not None:
            B, Q, T = bias.shape
            mask = (self.loc * bias).unsqueeze(1).expand(B, self.heads, Q, T)
            mask = mask.reshape(B * self.heads, Q, T)
        h, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv),
                         attn_mask=mask, need_weights=False)
        q = q + h
        return q + self.mlp(self.nm(q))


def token_grid_positions(n_tokens, device):
    """Normalized (u, v) in [-1, 1] of each token's centre, in the row-major order
    PatchEmbed produces (row = image x axis, col = image y axis)."""
    g = int(round(n_tokens ** 0.5))
    c = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) / g * 2 - 1
    uu, vv = torch.meshgrid(c, c, indexing='ij')
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)      # (T, 2)


class QueryProbe(nn.Module):
    """Truth-query probe.

    Two ingredients let a query actually reach its own particle's tokens:
      * each token carries its own grid position, embedded with the same Fourier
        basis the query uses;
      * the attention logits get an additive, learnable-strength locality prior
        -‖pos_query − pos_token‖² / 2σ².
    Without them the probe sits on a plateau where the attention stays diffuse
    and it only ever reproduces the event mean, i.e. its own floor. Both depend ONLY on the query's own position, so
    truth-anchoring is preserved exactly — permuting queries still permutes
    predictions (unit check U3).
    """

    def __init__(self, token_dim, n_feat, capacity='small', use_tokens=True,
                 n_fourier=None, pos_bias=True, sigma=0.25):
        super().__init__()
        cf = CAPACITY[capacity]
        d = cf['d_model']
        self.use_tokens = use_tokens
        self.pos_bias = pos_bias
        self.sigma = sigma
        self.n_fourier = N_FOURIER if n_fourier is None else n_fourier
        n_tok_feat = 2 + 4 * self.n_fourier
        self.tok_norm = nn.LayerNorm(token_dim)
        self.tok_proj = nn.Linear(token_dim, d)
        self.tok_pos = nn.Sequential(nn.Linear(n_tok_feat, d), nn.GELU(), nn.Linear(d, d))
        self.q_mlp = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([CrossBlock(d, cf['heads']) for _ in range(cf['layers'])])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self._pos_cache = None

    def _token_pos(self, tokens):
        if self._pos_cache is None or self._pos_cache.shape[0] != tokens.shape[1]:
            self._pos_cache = token_grid_positions(tokens.shape[1], tokens.device)
        return self._pos_cache

    def forward(self, tokens, qfeat, qpos=None):
        q = self.q_mlp(qfeat)
        kv = self.tok_proj(self.tok_norm(tokens.float()))
        if not self.use_tokens:
            kv = torch.zeros_like(kv)
        bias = None
        if self.use_tokens:
            tp = self._token_pos(tokens)                                   # (T, 2)
            tf = query_features(tp[:, 0], tp[:, 1], tp[:, 0], 1.0, 1.0, 0.0, 1.0,
                                False, self.n_fourier)                     # (T, F)
            kv = kv + self.tok_pos(tf).unsqueeze(0)
            if self.pos_bias and qpos is not None:
                d2 = ((qpos.unsqueeze(2) - tp.view(1, 1, -1, 2)) ** 2).sum(-1)
                bias = -d2 / (2 * self.sigma ** 2)
        for blk in self.blocks:
            q = blk(q, kv, bias)
        return self.head(q)
