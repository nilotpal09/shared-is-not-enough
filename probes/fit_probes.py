"""Fit readability probes on one encoder; optionally measure its movement.

    python probes/fit_probes.py --setting ijepa_shared_wider --seed 4
    python probes/fit_probes.py --setting byol_shared --seed 3 --key encoder     # online copy
    python probes/fit_probes.py --setting ijepa_shared_default --seed 1 --tag jepa-ep100
    python probes/fit_probes.py --untrained 5 --probe-n 100                       # untrained reference
    python probes/fit_probes.py --setting simclr_shared --seed 1 --kinds simple   # the small probe

Defaults are the paper's: the main probe ('big'), 40,000 steps, probe seeds 1
and 2, probe graininess N=1000, train/selection sets of 10,000/1,000 events.
Writes <out>/<arm>_s<seed>.npz (selection-set predictions and truth at the
best step, full held-out curves, movement per token if measured),
<out>/<arm>_s<seed>_probe.pt (best-step weights, scored on the test set by
score_test.py) and one row per fit to <out>/fits.csv.

Movement (--movement, at --movement-n, default N=1000 as in the paper) needs the GPU: the replicas come from the device's
random stream, so CPU replicas differ from the paper's GPU ones. The probe fit
itself agrees between CPU and GPU to about four decimals.
"""
import argparse
import csv
import os
import socket
import time
from pathlib import Path

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--setting', help='setting name (probes/runs.csv)')
g.add_argument('--untrained', type=int, metavar='SEED', help='random-init encoder from this seed')
ap.add_argument('--seed', type=int, help='training seed (with --setting)')
ap.add_argument('--tag', default=None, help='checkpoint tag (default: the final one, ep500)')
ap.add_argument('--key', default=None, choices=['target_encoder', 'encoder'],
                help="encoder copy (default: the one the paper probes)")
ap.add_argument('--probe-n', type=int, default=1000)
ap.add_argument('--steps', type=int, default=None, help='probe steps (default 40,000)')
ap.add_argument('--kinds', nargs='+', default=['big'], choices=['big', 'simple'])
ap.add_argument('--probe-seeds', type=int, nargs='+', default=[1, 2])
ap.add_argument('--n-train', type=int, default=None)
ap.add_argument('--movement', action='store_true', help='also measure movement')
ap.add_argument('--movement-n', type=int, default=1000, help='graininess of the movement replicas (the paper: 1000)')
ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
ap.add_argument('--threads', type=int, default=None, help='CPU threads')
ap.add_argument('--root', default=None, help='unzipped Zenodo record (default ./checkpoints)')
ap.add_argument('--cache', default=None, help='probe-set cache (default ./cache)')
ap.add_argument('--out', default='probe_fits', help='output folder')
args = ap.parse_args()

if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import common as C                                   # noqa: E402
import runs as R                                     # noqa: E402

C.set_device(args.device)
if args.threads:
    torch.set_num_threads(args.threads)
if args.movement:
    assert args.device == 'cuda', 'movement is measured on the GPU (see the module docstring)'
steps = args.steps or C.STEPS
n_train = args.n_train or C.N_TRAIN
n = args.probe_n

COLS = ['arm', 'setting', 'train_seed', 'run_dir', 'checkpoint_tag', 'encoder_key', 'probe', 'probe_seed',
        'probe_n', 'n_train_events', 'n_train_particles', 'n_eval_events', 'n_eval_particles',
        'n_probe_params', 'steps', 'best_step',
        'read_z1', 'read_z2', 'read_E', 'kn_z1', 'kn_z2', 'kn_E', 'delta_z1', 'delta_z2', 'delta_E',
        'read_z1_min', 'read_z2_min', 'read_E_min', 'step_z1_min', 'step_z2_min', 'step_E_min',
        'r_token_med', 'r_token_q1', 'r_token_q3', 'r_pooled',
        'sec_encoder', 'sec_resim', 'sec_probe', 'gpu_name', 'host', 'timestamp']


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, '') for c in COLS})


# ---- the encoder ---------------------------------------------------------
if args.untrained is not None:
    arm, setting, seed, run_dir = 'untrained', 'untrained', args.untrained, ''
    ckpt_tag, enc_key = 'random-init', 'random_init(no_checkpoint)'
    enc = C.untrained_encoder(seed)
else:
    assert args.seed is not None, '--seed is required with --setting'
    tags = sorted({r['checkpoint_tag'] for r in R.runs() if r['setting'] == args.setting
                   and r['seed'] == str(args.seed)}, key=lambda t: int(t.split('-ep')[1]))
    assert tags, f'no checkpoint for {args.setting} seed {args.seed} in probes/runs.csv'
    row = R.find(setting=args.setting, seed=args.seed, checkpoint_tag=args.tag or tags[-1])
    arm, setting, seed, run_dir, ckpt_tag = row['arm'], row['setting'], args.seed, row['run_folder'], row['checkpoint_tag']
    enc, enc_key = R.load(row, args.root, prefer=args.key)
tag = f'{arm.replace(" ", "-")}_s{seed}'
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
print(f'fit_probes | {setting} ({arm}) s{seed} {ckpt_tag} {enc_key} | N={n} | {steps} steps | '
      f'{args.kinds} x {args.probe_seeds} | {C.gpu_name()}', flush=True)

# ---- movement ------------------------------------------------------------
t0 = time.time()
res = None
if args.movement:
    nm = args.movement_n
    imgs = C.resim_images(C.make_resim_collator(nm), nm)
    res = C.per_token_resim(enc, imgs)
sec_resim = time.time() - t0

# ---- tokens --------------------------------------------------------------
t0 = time.time()
tr, ev = C.get_events(n, n_train=n_train, cache=args.cache)
kn = C.know_nothing(ev)
tok_tr = C.extract_tokens(enc, tr['imgs'], chunk=128)
tok_ev = C.extract_tokens(enc, ev['imgs'], chunk=128)
del enc
torch.cuda.empty_cache()
sec_tok = time.time() - t0

# ---- readability ---------------------------------------------------------
packed = dict(arm=arm, setting=setting, train_seed=seed, run_dir=run_dir, ckpt_tag=ckpt_tag,
              encoder_key=enc_key, probe_n=n, n_train_events=tr['imgs'].shape[0],
              n_eval_events=ev['imgs'].shape[0], n_eval_particles=int(ev['mask'].sum()),
              targets=np.array(C.TARGETS),
              know_nothing=np.array([kn['z1'], kn['z2'], kn['E']], dtype=np.float32),
              eval_x=ev['x'][ev['mask']].numpy(), eval_y=ev['y'][ev['mask']].numpy(),
              eval_lnE=ev['lnE'][ev['mask']].numpy(),
              eval_event=torch.nonzero(ev['mask'])[:, 0].numpy().astype(np.int32))
mov = {}
if res is not None:
    r, pooled, diag = res
    med, q1, q3 = C.r_summary(r)
    mov = dict(r_token_med=round(med, 6), r_token_q1=round(q1, 6), r_token_q3=round(q3, 6),
               r_pooled=round(pooled, 6))
    packed[f'r_token_N{nm}'] = r
    packed[f'r_pooled_N{nm}'] = np.float32(pooled)
    packed[f'intra_token_N{nm}'] = diag['intra_tok']
    packed[f'inter_token_N{nm}'] = diag['inter_tok']
weights = {}
for kind in args.kinds:
    for ps in args.probe_seeds:
        t0 = time.time()
        fit = C.fit_readability(tok_tr, tr, tok_ev, ev, probe_seed=ps, kind=kind, steps=steps,
                                half_w=C.HALF_W, half_h=C.HALF_H, device=C.DEVICE)
        sec_probe = time.time() - t0
        packed[f'pred_{kind}_ps{ps}'] = fit['pred']
        packed[f'curve_{kind}_ps{ps}'] = fit['curve']
        packed[f'best_step_{kind}_ps{ps}'] = np.int32(fit['step'])
        packed['truth'] = fit['truth']
        weights[f'{kind}_ps{ps}'] = fit['state']
        row = dict(arm=arm, setting=setting, train_seed=seed, run_dir=run_dir, checkpoint_tag=ckpt_tag,
                   encoder_key=enc_key, probe=kind, probe_seed=ps, probe_n=n,
                   n_train_events=tr['imgs'].shape[0], n_train_particles=int(tr['mask'].sum()),
                   n_eval_events=ev['imgs'].shape[0], n_eval_particles=int(ev['mask'].sum()),
                   n_probe_params=fit['n_params'], steps=steps, best_step=fit['step'],
                   sec_encoder=round(sec_tok, 1), sec_resim=round(sec_resim, 1),
                   sec_probe=round(sec_probe, 1), gpu_name=C.gpu_name(),
                   host=socket.gethostname(), timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'), **mov)
        for i, t in enumerate(C.TARGETS):
            row[f'read_{t}'] = round(fit['med'][i], 5)
            row[f'kn_{t}'] = round(kn[t], 5)
            row[f'delta_{t}'] = round(kn[t] - fit['med'][i], 5)
            row[f'read_{t}_min'] = round(fit['per_target_min'][i], 5)
            row[f'step_{t}_min'] = fit['per_target_argmin_step'][i]
        append_row(out / 'fits.csv', row)
        print(f'  {kind:<6s} ps{ps}  read z1 {fit["med"][0]:.4f} z2 {fit["med"][1]:.4f} '
              f'E {fit["med"][2]:.4f} @step {fit["step"]}  d(z1) {row["delta_z1"]:+.4f}  '
              f'[{sec_probe:.0f}s]', flush=True)

np.savez_compressed(out / f'{tag}.npz', **packed)
torch.save(weights, out / f'{tag}_probe.pt')
print(f'  -> {out / tag}.npz', flush=True)
