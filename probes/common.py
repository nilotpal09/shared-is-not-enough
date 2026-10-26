"""Shared probe machinery: probe images, targets, encoders, the probes, their
fit, and the movement measure.

Readability: median |prediction error| per particle for (z1, z2, E), read from
the frozen encoder's unpooled tokens by a truth-query probe (one query per
particle, built from its position only), at the selection-set best checkpoint,
against the constant-prediction ("know-nothing") error.

Movement: per-token ratio r = intra / inter. Replicas of the same truth that
redraw only shower shape (alpha, T; spots held fixed) give the intra distance;
different truths give the inter distance. Median over the 256 token positions.

The probe images, targets, fit and movement code are unchanged from the runs
behind the paper. Every probe set is generated from fixed seeds and cached
under the cache folder, so it is identical for every encoder.
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))       # the training code's src/ package

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import yaml                                                     # noqa: E402

from torch_calo_flash.calo_flash import get_longitudinal_parameters  # noqa: E402
from src.datasets.calo_flash import CaloFlashBatchCollator, CaloNorm  # noqa: E402
import src.models.vision_transformer as vit                     # noqa: E402

import probe_model as PM                                        # noqa: E402

# The pycaloflash GPU backend deposits spot energy with atomic scatter-adds, whose
# summation order is not fixed; without this, regenerating the same probe event
# twice moves ~6 cells in 10^6 by one fp16 ULP (measured). Determinism makes the
# probe set bit-identical for every encoder.
torch.use_deterministic_algorithms(True, warn_only=True)
# The fused/mem-efficient scaled-dot-product kernels have non-deterministic
# backwards; the math backend does not.
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

BASE_CONFIG = REPO / 'configs' / 'ijepa' / 'ijepa_shared_default.yml'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_root(root=None):
    """Folder holding the unzipped Zenodo record (final/, trajectory/, probes/, ...).
    Default: ./checkpoints beside the repository, or $TESTBED_ROOT."""
    return Path(root or os.environ.get('TESTBED_ROOT') or REPO / 'checkpoints')


def cache_root(cache=None):
    """Folder for the generated probe sets. Default: ./cache, or $TESTBED_CACHE."""
    return Path(cache or os.environ.get('TESTBED_CACHE') or REPO / 'cache')


def set_device(name):
    """'cuda' or 'cpu'. Functions read DEVICE at call time."""
    global DEVICE
    DEVICE = torch.device('cuda:0' if name == 'cuda' else 'cpu')
    return DEVICE


def gpu_name():
    try:
        return torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'cpu'
    except Exception:
        return 'cpu'


def base_cfg():
    """The simulator and encoder configuration every probe set is made with
    (all settings share the same calorimeter, generator, norm and ViT-nano)."""
    with open(BASE_CONFIG) as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------ constants

P_MAX = 20                  # generator N_range upper bound -> fixed slot count
PROBE_N = 1000              # default probe graininess (spots per layer)

# Pinned probe-set seeds. Fixed forever; changing them changes the probe set.
TRAIN_SEED_BASE = 20260804
EVAL_SEED_BASE = 20260805   # the selection set
TEST_SEED_BASE = 20260922   # the independent test set
GEN_CHUNK = 128             # events per simulator call

N_TRAIN = 10000
N_EVAL = 1000               # selection set
N_TEST = 1000

BATCH = 128
LR = 2e-3
STEPS = 40000               # the paper's probe budget
EVAL_EVERY = 100
PROBE_SEEDS = (1, 2)
# 'big': 85,189 params, 2 cross-attention blocks d64 (the main probe);
# 'simple': 7,811 params, one cross-attention pool + linear (the small probe).
PROBE_KINDS = ('big', 'simple')

# movement: M truths x K replicas, spots pinned, alpha/T redrawn per replica
M_TRUTHS = 64
K_REPLICAS = 8
RESIM_SEED_BASE = 42
FIXED_SPOT = 777

TARGETS = ('z1', 'z2', 'E')

HALF_W, HALF_H = 5.0, 5.0   # half-extent of the calorimeter face (cm); query u = x / HALF_W


# ------------------------------------------------------------------- encoders

def encoder_state_dict_from_ckpt(ckpt, prefer='target_encoder'):
    if prefer == 'target_encoder' and 'target_encoder' in ckpt:
        return {k.replace('module.', ''): v for k, v in ckpt['target_encoder'].items()}
    if 'encoder' in ckpt:
        return {k.replace('module.', ''): v for k, v in ckpt['encoder'].items()}
    if 'model' in ckpt:
        out = {}
        for k, v in ckpt['model'].items():
            if k.startswith('module.encoder.'):
                out[k[len('module.encoder.'):]] = v
            elif k.startswith('encoder.'):
                out[k[len('encoder.'):]] = v
        if out: return out
    raise ValueError(f'bad ckpt: {list(ckpt.keys())}')


def build_encoder(cfg, state_dict=None):
    factory = getattr(vit, cfg['meta']['model_name'])
    enc = factory(patch_size=cfg['mask']['patch_size'],
                  img_size=[cfg['data']['calo_cfg']['N_cells_x']],
                  in_chans=cfg['data']['calo_cfg']['N_cells_z']).to(DEVICE).eval()
    if state_dict is not None:
        enc.load_state_dict(state_dict, strict=True)
    return enc


def make_collator(calo_cfg, gen_cfg, n_spots, norm_cfg=None):
    cfg = {k: v for k, v in calo_cfg.items()
           if k not in ('N_spots_range', 'N_spots_context_range', 'N_spots_target_range')}
    cfg['N_spots_per_layer'] = int(n_spots)
    bc = CaloFlashBatchCollator(base_collator=lambda x: x, device=DEVICE,
                                 calo_cfg=cfg, generator_cfg=gen_cfg,
                                 norm=CaloNorm(**(norm_cfg or {})))
    bc._build_backend()
    return bc


def load_encoder(ckpt_path, cfg=None, prefer='target_encoder'):
    """Encoder from a training checkpoint. Returns (encoder, key) where key
    records the weights actually read: 'target_encoder' (EMA copy) when asked for
    and present, else 'encoder' (online copy), else 'model(online)' (MAE*)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    key = prefer if (prefer == 'target_encoder' and 'target_encoder' in ckpt) \
        else ('encoder' if 'encoder' in ckpt else 'model(online)')
    sd = encoder_state_dict_from_ckpt(ckpt, prefer=prefer)
    torch.manual_seed(0)
    enc = build_encoder(cfg or base_cfg(), sd)
    del ckpt, sd
    return enc.eval(), key


def untrained_encoder(seed, cfg=None):
    """Random-init encoder of the same architecture, reproducible from its seed."""
    torch.manual_seed(int(seed))
    return build_encoder(cfg or base_cfg(), None).eval()


@torch.no_grad()
def extract_tokens(enc, imgs_cpu, chunk=128, out_dtype=torch.float16):
    """Unpooled token grid: (B, 256, 96) for vit_nano/patch-4 on 64x64x4."""
    out = []
    for i in range(0, imgs_cpu.size(0), chunk):
        x = imgs_cpu[i:i + chunk].to(DEVICE, non_blocking=True).float()
        out.append(enc(x).to(out_dtype))
        del x
    return torch.cat(out, dim=0)


# ------------------------------------------------ probe events + targets

def _chunk_seeds(base, c):
    ts = (base + 7919 * c) & 0x7FFFFFFF
    ls = (ts * 104729 + 17) & 0x7FFFFFFF
    ss = (ts * 31337 + 101) & 0x7FFFFFFF
    return ts, ls, ss


@torch.no_grad()
def gen_probe_events(coll, n_events, seed_base, n_spots=PROBE_N, chunk=GEN_CHUNK, verbose=False):
    """Fresh pycaloflash events with PINNED seeds.

    Returns dict of CPU tensors:
        imgs      (N, 4, 64, 64)  float16, model-input normalized
        x, y      (N, P_MAX)      float32, particle positions (cm), NaN-free (0 where pad)
        lnE       (N, P_MAX)      float32
        ln_alpha  (N, P_MAX)      float32   realized, as used by the simulator
        ln_T      (N, P_MAX)      float32   realized
        mask      (N, P_MAX)      bool      True = real particle
    """
    imgs, X, Y, LE, LA, LT, MK = [], [], [], [], [], [], []
    done = 0
    c = 0
    while done < n_events:
        b = min(chunk, n_events - done)
        ts, ls, ss = _chunk_seeds(seed_base, c)
        p_E, p_x, p_y = coll._generate_truth(b, ts)
        lp = coll._compute_long_params(p_E, ls)
        img = coll.norm.forward(coll._simulate_once(
            p_E, p_x, p_y, ss, n_spots_override=n_spots,
            long_params_override=lp))

        P = p_E.shape[1]
        flat = p_E.reshape(-1)
        real = torch.where(flat == flat)[0]
        ev, slot = real // P, real % P
        z = lambda: torch.zeros(b, P_MAX, device=p_E.device, dtype=torch.float32)
        xg, yg, leg, lag, ltg = z(), z(), z(), z(), z()
        mg = torch.zeros(b, P_MAX, device=p_E.device, dtype=torch.bool)
        assert P <= P_MAX, f'P={P} > P_MAX={P_MAX}'
        xg[ev, slot] = p_x.reshape(-1)[real].float()
        yg[ev, slot] = p_y.reshape(-1)[real].float()
        leg[ev, slot] = torch.log(flat[real].float())
        lag[ev, slot] = torch.log(lp['alpha'].float())
        ltg[ev, slot] = torch.log(lp['T'].float())
        mg[ev, slot] = True

        imgs.append(img.to(torch.float16).cpu())
        X.append(xg.cpu()); Y.append(yg.cpu()); LE.append(leg.cpu())
        LA.append(lag.cpu()); LT.append(ltg.cpu()); MK.append(mg.cpu())
        done += b
        c += 1
        if verbose and c % 20 == 0:
            print(f'    generated {done}/{n_events}', flush=True)
    return dict(imgs=torch.cat(imgs), x=torch.cat(X), y=torch.cat(Y),
                lnE=torch.cat(LE), ln_alpha=torch.cat(LA), ln_T=torch.cat(LT),
                mask=torch.cat(MK))


def make_probe_collator(n_spots=PROBE_N, cfg=None):
    cfg = cfg or base_cfg()
    return make_collator(cfg['data']['calo_cfg'], cfg['data'].get('generator_cfg', {}),
                         n_spots, cfg['data'].get('norm') or {})


def z_targets(data, Z, lnE_mu=None, lnE_sd=None):
    """Exact (z1, z2) inversion + standardized lnE, per particle.

        A  = (ln alpha - mu_alpha) / sigma_alpha
        Tt = (ln T     - mu_T)     / sigma_T
        z1 = (Tt + A) / (sqrt(2) a),  a = sqrt(1 + rho)
        z2 = (Tt - A) / (sqrt(2) b),  b = sqrt(1 - rho)

    mu/sigma/rho are deterministic functions of the particle energy, so they are
    recomputed here from E (the z draws inside get_longitudinal_parameters are
    discarded). Pads are zeroed: they are masked out everywhere, but unzeroed
    NaNs poison the loss.

    Returns (targets (N, P, 3) float32, lnE_mu, lnE_sd).
    """
    m = data['mask']
    E = torch.exp(data['lnE'][m].float()).clamp(min=1e-12)
    with torch.random.fork_rng(devices=[]):      # the z draws inside are discarded
        lp = get_longitudinal_parameters(E.cpu(), int(Z))

    A = (data['ln_alpha'][m] - lp['mean_ln_alpha']) / lp['sigma_ln_alpha']
    Tt = (data['ln_T'][m] - lp['mean_ln_T']) / lp['sigma_ln_T']
    rho = lp['rho_ln_T_ln_alpha']
    a = torch.sqrt((1 + rho).clamp(min=0))
    b = torch.sqrt((1 - rho).clamp(min=0))

    if lnE_mu is None:
        lnE_mu = float(data['lnE'][m].mean())
        lnE_sd = float(data['lnE'][m].std())

    cols = [(Tt + A) / (math.sqrt(2.0) * a),
            (Tt - A) / (math.sqrt(2.0) * b),
            (data['lnE'][m] - lnE_mu) / lnE_sd]
    t = torch.zeros(*data['lnE'].shape, 3, dtype=torch.float32)   # pads stay 0
    for i, c in enumerate(cols):
        t[..., i].masked_scatter_(m, c.float())
    assert torch.isfinite(t).all(), 'non-finite z target'
    return t, lnE_mu, lnE_sd


def probe_events(n_train=N_TRAIN, n_eval=N_EVAL, n_spots=PROBE_N, verbose=False):
    """Pinned probe events + z targets. Identical for every encoder.

    The train set is generated with TRAIN_SEED_BASE in the same chunked seed
    schedule for any n_train; the selection set with EVAL_SEED_BASE.
    """
    coll = make_probe_collator(n_spots)
    Z = base_cfg()['data']['calo_cfg']['Z']
    tr = gen_probe_events(coll, n_train, TRAIN_SEED_BASE, n_spots, verbose=verbose)
    ev = gen_probe_events(coll, n_eval, EVAL_SEED_BASE, n_spots, verbose=verbose)
    del coll
    torch.cuda.empty_cache()
    tr['t3'], mu, sd = z_targets(tr, Z)
    ev['t3'], _, _ = z_targets(ev, Z, mu, sd)
    tr['lnE_mu'], tr['lnE_sd'] = mu, sd
    return tr, ev


def get_events(n_spots=PROBE_N, n_train=N_TRAIN, n_eval=N_EVAL, cache=None, verbose=False):
    """(train, selection) probe sets at this graininess, from the cache or made
    and cached on first use."""
    f = cache_root(cache) / f'probeN{n_spots}' / f'probe_events_{n_train}_{n_eval}.pt'
    if f.exists():
        d = torch.load(f, map_location='cpu', weights_only=False)
        return d['tr'], d['ev']
    tr, ev = probe_events(n_train, n_eval, n_spots, verbose=verbose)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix('.tmp')
    torch.save(dict(tr=tr, ev=ev), tmp)
    tmp.rename(f)
    return tr, ev


def get_test_events(n_spots=PROBE_N, cache=None):
    """The independent test set at this graininess (TEST_SEED_BASE; z targets use
    the train set's lnE mean/sd, as for the selection set)."""
    f = cache_root(cache) / 'test' / f'probeN{n_spots}' / f'probe_events_test_{N_TEST}.pt'
    if f.exists():
        te = torch.load(f, map_location='cpu', weights_only=False)
    else:
        tr, _ = get_events(n_spots, cache=cache)
        coll = make_probe_collator(n_spots)
        te = gen_probe_events(coll, N_TEST, TEST_SEED_BASE, n_spots)
        del coll
        torch.cuda.empty_cache()
        te['t3'], _, _ = z_targets(te, base_cfg()['data']['calo_cfg']['Z'], tr['lnE_mu'], tr['lnE_sd'])
        te['seed_base'] = TEST_SEED_BASE
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix('.tmp')
        torch.save(te, tmp)
        tmp.rename(f)
    assert int(te['seed_base']) == TEST_SEED_BASE
    return te


def know_nothing(ev):
    """Median |error| of the constant predictor (the eval-set median)."""
    m = ev['mask']
    out = {}
    for i, k in enumerate(TARGETS):
        v = ev['t3'][..., i][m]
        out[k] = float((v - v.median()).abs().median())
    return out


# ------------------------------------------------------------------ probes

def _pos_feats(u, v):
    """The 6 position-only query features: u, v and one Fourier octave of each.
    NO energy — the energy shortcut is the contamination this rerun removes."""
    return torch.stack([u, v,
                        torch.sin(math.pi * u), torch.cos(math.pi * u),
                        torch.sin(math.pi * v), torch.cos(math.pi * v)], dim=-1)


def query_feats(x, y, half_w, half_h):
    u = (x / half_w).clamp(-2, 2)
    v = (y / half_h).clamp(-2, 2)
    return _pos_feats(u, v), torch.stack([u, v], dim=-1)


def token_pos_feats(n_tokens, device):
    """Token-centre (u, v) in [-1, 1], row-major as PatchEmbed emits them."""
    g = int(round(n_tokens ** 0.5))
    c = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) / g * 2 - 1
    uu, vv = torch.meshgrid(c, c, indexing='ij')
    return _pos_feats(uu.reshape(-1), vv.reshape(-1))


class SimpleProbe(nn.Module):
    """Truth-query cross-attention pool + linear. 7,811 parameters.

    One query per TRUE particle, built ONLY from that particle's known position.
    The query cross-attends (single head, no query-to-query attention) over the
    frozen encoder's unpooled token grid and a linear head reads (z1, z2, E) off
    the pooled token. Truth-anchored by construction: permuting the queries
    permutes the predictions exactly.
    """

    def __init__(self, token_dim=96, d=64, n_out=3, n_feat=6):
        super().__init__()
        self.tok_norm = nn.LayerNorm(token_dim)       #   192
        self.tok_pos = nn.Linear(n_feat, token_dim)   #   672
        self.k_proj = nn.Linear(token_dim, d)         # 6,208
        self.q_proj = nn.Linear(n_feat, d)            #   448
        self.head = nn.Linear(token_dim, n_out)       #   291
        self.scale = d ** -0.5
        self._pos = None

    def forward(self, tokens, qfeat, qpos=None):
        # tokens (B, T, D) ; qfeat (B, P, 6)
        if self._pos is None or self._pos.shape[0] != tokens.shape[1]:
            self._pos = token_pos_feats(tokens.shape[1], tokens.device)
        kv = self.tok_norm(tokens.float()) + self.tok_pos(self._pos).unsqueeze(0)
        k = self.k_proj(kv)                                        # (B, T, d)
        q = self.q_proj(qfeat)                                     # (B, P, d)
        att = torch.softmax(torch.einsum('bpd,btd->bpt', q, k) * self.scale, dim=-1)
        pooled = torch.einsum('bpt,btd->bpd', att, kv)             # (B, P, D)
        return self.head(pooled)


class BigProbe(PM.QueryProbe):
    """The repo's truth-query probe (probe_model.QueryProbe, capacity 'big')
    with a 3-output head: 85,189 parameters, 22-dim position-only queries,
    Fourier token positions, learnable locality prior sigma=0.25.

    This is the probe whose numbers the twocase table records.
    """

    def __init__(self, token_dim=96, n_feat=22, n_out=3):
        super().__init__(token_dim, n_feat, capacity='big')
        d = PM.CAPACITY['big']['d_model']
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))


N_PARAMS = {'big': 85189, 'simple': 7811, 'z1big': 85059}


def build_probe(kind, token_dim=96):
    if kind == 'big':
        return BigProbe(token_dim)
    if kind == 'simple':
        return SimpleProbe(token_dim)
    if kind == 'z1big':                 # the single-target probe: z1 only
        return BigProbe(token_dim, n_out=1)
    raise KeyError(f'unknown probe {kind!r}; known: {PROBE_KINDS + ("z1big",)}')


def probe_query_feats(kind, x, y, half_w, half_h):
    """(query features, query positions) for a probe rung. Position only."""
    if kind == 'simple':
        return query_feats(x, y, half_w, half_h)
    qf = PM.query_features(x, y, x, half_w, half_h, 0.0, 1.0, use_energy=False)
    qp = torch.stack([(x / half_w).clamp(-2, 2), (y / half_h).clamp(-2, 2)], dim=-1)
    return qf, qp


def _prep(data, tok, device, half_w, half_h, kind):
    m = data['mask'].to(device)
    qf, qp = probe_query_feats(kind, data['x'].to(device), data['y'].to(device),
                               half_w, half_h)
    return dict(mask=m, qf=qf, qpos=qp, t3=data['t3'].to(device), tok=tok)


@torch.no_grad()
def _eval_preds(model, ev, chunk=256):
    model.eval()
    P, T = [], []
    for i in range(0, ev['mask'].shape[0], chunk):
        sl = slice(i, i + chunk)
        p = model(ev['tok'][sl], ev['qf'][sl], ev['qpos'][sl])
        m = ev['mask'][sl]
        P.append(p[m].float())
        T.append(ev['t3'][sl][m].float())
    model.train()
    return torch.cat(P), torch.cat(T)


def fit_readability(tok_tr, data_tr, tok_ev, data_ev, *, probe_seed, kind='big',
                    steps=STEPS, batch=BATCH, lr=LR, eval_every=EVAL_EVERY,
                    half_w=5.0, half_h=5.0, device=None, verbose=False):
    """Train one probe; return best-checkpoint predictions, weights and curve.

    Held-out is evaluated every `eval_every` steps; the reported checkpoint is
    the held-out minimum of the mean of the three median |error| values. The
    FULL curve is returned so overtraining is visible per encoder instead of
    assumed handled.
    """
    device = device or DEVICE
    torch.manual_seed(int(probe_seed))
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))

    tr = _prep(data_tr, tok_tr, device, half_w, half_h, kind)
    ev = _prep(data_ev, tok_ev, device, half_w, half_h, kind)

    token_dim = tok_tr.shape[-1]
    model = build_probe(kind, token_dim=token_dim).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    if token_dim == 96:      # the house encoder width; wider encoders (ViT-Tiny,
        assert n_par == N_PARAMS[kind], \
            f'{kind} probe is {n_par}p, expected {N_PARAMS[kind]}'
    # ViT-Small) necessarily give the probe a wider input layer, so its size is
    # recorded per row instead of asserted.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = tr['mask'].shape[0]
    curve = []                       # (step, med_z1, med_z2, med_E)
    best = dict(score=float('inf'))
    perm = torch.randperm(N, generator=g)
    ptr = 0
    model.train()
    for s in range(steps):
        if ptr + batch > N:
            perm = torch.randperm(N, generator=g)
            ptr = 0
        sel = perm[ptr:ptr + batch].to(device)
        ptr += batch
        pred = model(tr['tok'][sel], tr['qf'][sel], tr['qpos'][sel])
        m = tr['mask'][sel]
        loss = F.mse_loss(pred[m], tr['t3'][sel][m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % eval_every == 0 or s == steps - 1:
            p, t = _eval_preds(model, ev)
            med = (p - t).abs().median(dim=0).values.cpu().numpy()
            curve.append((s + 1, *[float(x) for x in med]))
            score = float(med.mean())
            if score < best['score']:
                best = dict(score=score, step=s + 1,
                            med=[float(x) for x in med],
                            pred=p.cpu().numpy().astype(np.float32),
                            truth=t.cpu().numpy().astype(np.float32),
                            state={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()})
            if verbose:
                print(f'      step {s+1:5d} loss {float(loss):.4f}  '
                      f'z1 {med[0]:.4f} z2 {med[1]:.4f} E {med[2]:.4f}', flush=True)

    cur = np.array(curve, dtype=np.float32)
    best['curve'] = cur
    best['per_target_min'] = [float(cur[:, i + 1].min()) for i in range(3)]
    best['per_target_argmin_step'] = [int(cur[int(cur[:, i + 1].argmin()), 0])
                                      for i in range(3)]
    best['n_params'] = int(n_par)
    best['kind'] = kind
    return best


def fit_z1(tok_tr, data_tr, tok_ev, data_ev, tok_te, data_te, *, probe_seed, steps=STEPS,
           batch=BATCH, lr=LR, eval_every=EVAL_EVERY, device=None):
    """The single-target probe: fit_readability restricted to z1 (BigProbe trunk,
    1-output head), with the test set evaluated at every checkpoint. The
    checkpoint is chosen on the selection-set z1 median |error|; the test-set
    value at that checkpoint is returned alongside."""
    device = device or DEVICE
    torch.manual_seed(int(probe_seed))
    g = torch.Generator(device='cpu').manual_seed(int(probe_seed))

    def prep(data, tok):
        m = data['mask'].to(device)
        qf, qp = probe_query_feats('big', data['x'].to(device),
                                     data['y'].to(device), HALF_W, HALF_H)
        return dict(mask=m, qf=qf, qpos=qp, t1=data['t3'].to(device)[..., :1], tok=tok)

    tr, ev, te = prep(data_tr, tok_tr), prep(data_ev, tok_ev), prep(data_te, tok_te)
    token_dim = tok_tr.shape[-1]
    model = BigProbe(token_dim=token_dim, n_out=1).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    if token_dim == 96:
        assert n_par == N_PARAMS['z1big'], f'z1big probe is {n_par}p'
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def eval_preds(dd, chunk=256):
        model.eval()
        P, T = [], []
        for i in range(0, dd['mask'].shape[0], chunk):
            sl = slice(i, i + chunk)
            p = model(dd['tok'][sl], dd['qf'][sl], dd['qpos'][sl])
            m = dd['mask'][sl]
            P.append(p[m].float())
            T.append(dd['t1'][sl][m].float())
        model.train()
        return torch.cat(P), torch.cat(T)

    N = tr['mask'].shape[0]
    curve = []
    best = dict(score=float('inf'))
    perm = torch.randperm(N, generator=g)
    ptr = 0
    model.train()
    for s in range(steps):
        if ptr + batch > N:
            perm = torch.randperm(N, generator=g)
            ptr = 0
        sel = perm[ptr:ptr + batch].to(device)
        ptr += batch
        pred = model(tr['tok'][sel], tr['qf'][sel], tr['qpos'][sel])
        m = tr['mask'][sel]
        loss = F.mse_loss(pred[m], tr['t1'][sel][m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % eval_every == 0 or s == steps - 1:
            p, t = eval_preds(ev)
            med = float((p - t).abs().median())
            p2, t2 = eval_preds(te)
            med_te = float((p2 - t2).abs().median())
            curve.append((s + 1, med, med_te))
            if med < best['score']:
                best = dict(score=med, step=s + 1, med=med, med_test=med_te,
                            pred=p.cpu().numpy().astype(np.float32)[:, 0],
                            truth=t.cpu().numpy().astype(np.float32)[:, 0],
                            pred_test=p2.cpu().numpy().astype(np.float32)[:, 0],
                            truth_test=t2.cpu().numpy().astype(np.float32)[:, 0],
                            state={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()})
    cur = np.array(curve, dtype=np.float32)
    best['curve'] = cur
    best['min'] = float(cur[:, 1].min())
    best['argmin_step'] = int(cur[int(cur[:, 1].argmin()), 0])
    best['n_params'] = int(n_par)
    return best


# ------------------------------------------------------------ movement

def make_resim_collator(n_spots):
    """A collator pinned at `n_spots` for the movement replicas."""
    return make_probe_collator(int(n_spots))


def resim_images(coll, n_spots):
    """The alpha/T-only channel, verbatim archive conventions: spots CRN-pinned
    (fixed_spot_seed), alpha/T re-rolled per replica, M truths x K replicas."""
    _, reps = coll.generate_replicas(batch_size=M_TRUTHS, n_replicas=K_REPLICAS,
                                     seed=RESIM_SEED_BASE + int(n_spots),
                                     normalize=True, fixed_spot_seed=FIXED_SPOT)
    return reps.reshape(M_TRUTHS * K_REPLICAS, *reps.shape[2:]).float().cpu()


@torch.no_grad()
def _encode_tokens(enc, imgs_cpu, chunk=64):
    out = []
    for i in range(0, imgs_cpu.size(0), chunk):
        out.append(enc(imgs_cpu[i:i + chunk].to(DEVICE, non_blocking=True).float()).float())
    return torch.cat(out, dim=0)


def _intra_inter(z):
    """z: (M, K, ..., D) L2-normalized on the last axis. Returns (intra, inter)
    with any middle axes kept."""
    M, K = z.shape[0], z.shape[1]
    iu, ju = torch.triu_indices(K, K, offset=1)
    intra = (1.0 - (z[:, iu] * z[:, ju]).sum(-1)).mean(dim=(0, 1))
    z0 = z[:, 0]
    iU, jU = torch.triu_indices(M, M, offset=1)
    inter = (1.0 - (z0[iU] * z0[jU]).sum(-1)).mean(dim=0)
    return intra, inter


@torch.no_grad()
def per_token_resim(enc, imgs):
    """Returns (r_per_token (T,) numpy, pooled r float, intra/inter diagnostics).

    Pooled path is the archive's exactly: mean over tokens, then L2-normalize,
    then intra/inter. Per-token path skips the mean-pool: every token position
    is normalized on its own and gets its own intra/inter ratio.
    """
    h = _encode_tokens(enc, imgs)                                  # (M*K, T, D)
    T = h.shape[1]

    zp = F.normalize(h.mean(dim=1), dim=-1).reshape(M_TRUTHS, K_REPLICAS, -1)
    ip, ep = _intra_inter(zp)
    pooled = float(ip) / (float(ep) + 1e-12)

    zt = F.normalize(h, dim=-1).reshape(M_TRUTHS, K_REPLICAS, T, -1)
    it, et = _intra_inter(zt)
    r = (it / (et + 1e-12)).cpu().numpy().astype(np.float32)
    return r, pooled, dict(intra_pooled=float(ip), inter_pooled=float(ep),
                           intra_tok=it.cpu().numpy().astype(np.float32),
                           inter_tok=et.cpu().numpy().astype(np.float32))


def r_summary(r):
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    return float(med), float(q1), float(q3)
