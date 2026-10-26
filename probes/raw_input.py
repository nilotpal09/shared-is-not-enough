"""The main probe on the raw input patches, no encoder.

Same fit as every main-probe score (fit_readability, 'big', 40,000 steps,
selection every 100 steps, best mean MedAE over z1/z2/E), then the selected
checkpoint is scored on the test set. Tokens are the model-input image cut into
the encoder's 4x4 patches in the encoder's token order: (B, 256, 4*4*4 = 64).

Variants:
  raw_ln     the probe as it is (its per-token LayerNorm removes each patch's
             overall level; for raw pixels that also removes how intensity
             changes from patch to patch)
  raw_noln   the same probe with that LayerNorm replaced by a fixed
             per-dimension standardisation computed on the fitting set
  untrained  a random-init encoder (seed --init) through this same script

    python probes/raw_input.py --variant raw_ln --probe-seed 1
"""
import argparse
import csv
import socket
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True, choices=['raw_ln', 'raw_noln', 'untrained'])
ap.add_argument('--probe-seed', type=int, required=True)
ap.add_argument('--init', type=int, default=1)
ap.add_argument('--steps', type=int, default=40000)
ap.add_argument('--cache', default=None)
ap.add_argument('--out', default='raw_input')
args = ap.parse_args()

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn as nn                                # noqa: E402

import common as C                                   # noqa: E402


def patchify(imgs, chunk=512):
    out = []
    for i in range(0, imgs.shape[0], chunk):
        x = imgs[i:i + chunk].float()                       # (b, 4, 64, 64)
        b = x.shape[0]
        x = x.reshape(b, 4, 16, 4, 16, 4).permute(0, 2, 4, 1, 3, 5)   # (b, 16, 16, 4, 4, 4)
        out.append(x.reshape(b, 256, 64).to(torch.float16))
    return torch.cat(out)


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tr, ev = C.get_events(cache=args.cache)
te = C.get_test_events(cache=args.cache)
kn, kn_te = C.know_nothing(ev), C.know_nothing(te)
t0 = time.time()
if args.variant == 'untrained':
    enc = C.untrained_encoder(args.init).to(C.DEVICE)
    tok_tr, tok_ev, tok_te = (C.extract_tokens(enc, d['imgs'], chunk=256) for d in (tr, ev, te))
    del enc
else:
    tok_tr, tok_ev, tok_te = (patchify(d['imgs']) for d in (tr, ev, te))
    if args.variant == 'raw_noln':
        mu = tok_tr.float().mean(dim=(0, 1), keepdim=True)
        sd = tok_tr.float().std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        tok_tr, tok_ev, tok_te = (((t.float() - mu) / sd).to(torch.float16)
                                  for t in (tok_tr, tok_ev, tok_te))
        _build = C.build_probe

        def build_noln(kind, token_dim=96):
            m = _build(kind, token_dim)
            m.tok_norm = nn.Identity()
            return m
        C.build_probe = build_noln          # fit_readability builds its probe through this
tok_tr, tok_ev, tok_te = (t.to(C.DEVICE) for t in (tok_tr, tok_ev, tok_te))
sec_tok = time.time() - t0
t0 = time.time()
fit = C.fit_readability(tok_tr, tr, tok_ev, ev, probe_seed=args.probe_seed, kind='big',
                        steps=args.steps, half_w=C.HALF_W, half_h=C.HALF_H)
sec_probe = time.time() - t0
# test-set score of the selected checkpoint
model = C.build_probe('big', token_dim=tok_tr.shape[-1]).to(C.DEVICE)
model.load_state_dict(fit['state'])
p, t = C._eval_preds(model, C._prep(te, tok_te, C.DEVICE, C.HALF_W, C.HALF_H, 'big'))
med_te = (p - t).abs().median(dim=0).values.cpu().numpy()
p2, t2 = C._eval_preds(model, C._prep(ev, tok_ev, C.DEVICE, C.HALF_W, C.HALF_H, 'big'))
med_sel = (p2 - t2).abs().median(dim=0).values.cpu().numpy()
assert np.allclose(med_sel, fit['med'], atol=2e-3), (med_sel, fit['med'])   # loading check
row = dict(variant=args.variant, init=args.init if args.variant == 'untrained' else '',
           probe_seed=args.probe_seed, steps=args.steps, best_step=fit['step'],
           n_probe_params=fit['n_params'], token_dim=tok_tr.shape[-1],
           gpu=C.gpu_name(), host=socket.gethostname(),
           sec_tokens=round(sec_tok, 1), sec_probe=round(sec_probe, 1),
           timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))
for i, tg in enumerate(C.TARGETS):
    row[f'delta_{tg}_sel'] = round(kn[tg] - fit['med'][i], 5)
    row[f'delta_{tg}_test'] = round(kn_te[tg] - float(med_te[i]), 5)
out_csv = out / 'raw_input.csv'
new = not out_csv.exists()
with open(out_csv, 'a', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
tag = f'{args.variant}' + (f'_i{args.init}' if args.variant == 'untrained' else '') + f'_ps{args.probe_seed}'
np.savez_compressed(out / f'{tag}.npz', curve=fit['curve'], pred=fit['pred'], truth=fit['truth'],
                    pred_test=p.cpu().numpy(), truth_test=t.cpu().numpy())
print(f'{tag}: d(z1) sel {row["delta_z1_sel"]:+.4f} test {row["delta_z1_test"]:+.4f} | '
      f'd(E) test {row["delta_E_test"]:+.4f} | best step {fit["step"]} | {fit["n_params"]}p', flush=True)
