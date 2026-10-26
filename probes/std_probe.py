"""Main probe with its per-token LayerNorm replaced by a fixed per-dimension
standardisation (computed on the fitting set), on trained and untrained encoders.

The same change raised the raw-patch score from 0.322 to 0.396 (raw_input.py,
variant raw_noln). This asks whether the probe's LayerNorm is what keeps trained
encoders below the untrained one. Everything else as the paper's main probe:
fit_readability, 'big', 40,000 steps, N = 1000, selected checkpoint scored on
the test set. Its scores are in results/standardised_probe.csv; the LayerNorm
scores to compare with are in results/readability.csv.

    python probes/std_probe.py --setting ijepa_shared_wider --seed 1 --probe-seed 1
    python probes/std_probe.py --untrained 1 --probe-seed 1
"""
import argparse
import csv
import socket
import time
from pathlib import Path

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--setting')
g.add_argument('--untrained', type=int, metavar='SEED')
ap.add_argument('--seed', type=int)
ap.add_argument('--probe-seed', type=int, default=1)
ap.add_argument('--steps', type=int, default=40000)
ap.add_argument('--root', default=None, help='unzipped Zenodo record (default ./checkpoints)')
ap.add_argument('--cache', default=None, help='probe-set cache (default ./cache)')
ap.add_argument('--out', default='std_probe', help='output folder')
args = ap.parse_args()

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn as nn                                # noqa: E402

import common as C                                   # noqa: E402
import runs as R                                     # noqa: E402

if args.untrained is not None:
    setting, seed, tag_ck = 'untrained', args.untrained, 'random-init'
    enc = C.untrained_encoder(seed).to(C.DEVICE)
else:
    row = R.find(setting=args.setting, seed=args.seed, checkpoint_tag='jepa-ep500'
                 if not args.setting.startswith('mae') else 'mae-ep500')
    setting, seed, tag_ck = args.setting, args.seed, row['checkpoint_tag']
    enc, _ = R.load(row, args.root)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

tr, ev = C.get_events(cache=args.cache)
te = C.get_test_events(cache=args.cache)
kn, kn_te = C.know_nothing(ev), C.know_nothing(te)
t0 = time.time()
tok_tr, tok_ev, tok_te = (C.extract_tokens(enc, d['imgs'], chunk=128) for d in (tr, ev, te))
del enc
torch.cuda.empty_cache()
mu = tok_tr.float().mean(dim=(0, 1), keepdim=True)
sd = tok_tr.float().std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
tok_tr, tok_ev, tok_te = (((t.float() - mu) / sd).to(torch.float16) for t in (tok_tr, tok_ev, tok_te))
sec_tok = time.time() - t0

_build = C.build_probe


def build_noln(kind, token_dim=96):
    m = _build(kind, token_dim)
    m.tok_norm = nn.Identity()
    return m


C.build_probe = build_noln          # fit_readability builds its probe through this
C.N_PARAMS = dict(C.N_PARAMS, big=C.N_PARAMS["big"] - 2 * 96)   # minus the LayerNorm(96) weight and bias

t0 = time.time()
fit = C.fit_readability(tok_tr, tr, tok_ev, ev, probe_seed=args.probe_seed, kind='big',
                        steps=args.steps, half_w=C.HALF_W, half_h=C.HALF_H)
sec_probe = time.time() - t0
model = C.build_probe('big', token_dim=tok_tr.shape[-1]).to(C.DEVICE)
model.load_state_dict(fit['state'])
p, t = C._eval_preds(model, C._prep(te, tok_te, C.DEVICE, C.HALF_W, C.HALF_H, 'big'))
med_te = (p - t).abs().median(dim=0).values.cpu().numpy()
p2, t2 = C._eval_preds(model, C._prep(ev, tok_ev, C.DEVICE, C.HALF_W, C.HALF_H, 'big'))
med_sel = (p2 - t2).abs().median(dim=0).values.cpu().numpy()
assert np.allclose(med_sel, fit['med'], atol=2e-3), (med_sel, fit['med'])   # loading check
row = dict(setting=setting, train_seed=seed, checkpoint_tag=tag_ck, probe_seed=args.probe_seed,
           steps=args.steps, best_step=fit['step'], n_probe_params=fit['n_params'],
           gpu=C.gpu_name(), host=socket.gethostname(),
           sec_tokens=round(sec_tok, 1), sec_probe=round(sec_probe, 1),
           timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))
for i, tg in enumerate(C.TARGETS):
    row[f'delta_{tg}_sel'] = round(kn[tg] - fit['med'][i], 5)
    row[f'delta_{tg}_test'] = round(kn_te[tg] - float(med_te[i]), 5)
out_csv = out / 'std_probe.csv'
new = not out_csv.exists()
with open(out_csv, 'a', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
name = f'{setting}_s{seed}_ps{args.probe_seed}'
np.savez_compressed(out / f'{name}.npz', curve=fit['curve'], pred=fit['pred'], truth=fit['truth'],
                    pred_test=p.cpu().numpy(), truth_test=t.cpu().numpy())
print(f'{name}: d(z1) sel {row["delta_z1_sel"]:+.4f} test {row["delta_z1_test"]:+.4f} | '
      f'd(E) test {row["delta_E_test"]:+.4f} | best step {fit["step"]} | {sec_probe:.0f}s', flush=True)
