"""The single-target probe: the main probe's trunk with a z1-only head, fitted
with the selection set choosing the step and the test set scored at it.

    python probes/fit_z1.py --setting ijepa_shared_default --seed 3
    python probes/fit_z1.py --setting byol_shared --seed 3 --key encoder
    python probes/fit_z1.py --untrained 4

Writes <out>/<arm>_s<seed>_st<steps>.npz (selection and test predictions and
truth at the best step, the curve), the weights beside it (z1big_ps<k>), and
one row per fit to <out>/fits_z1.csv.
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
ap.add_argument('--tag', default=None)
ap.add_argument('--key', default=None, choices=['target_encoder', 'encoder'])
ap.add_argument('--steps', type=int, default=None, help='probe steps (default 40,000)')
ap.add_argument('--probe-seeds', type=int, nargs='+', default=[1, 2])
ap.add_argument('--root', default=None)
ap.add_argument('--cache', default=None)
ap.add_argument('--out', default='probe_fits_z1')
args = ap.parse_args()

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import common as C                                   # noqa: E402
import runs as R                                     # noqa: E402

steps = args.steps or C.STEPS
COLS = ['arm', 'setting', 'train_seed', 'run_dir', 'checkpoint_tag', 'encoder_key', 'probe', 'probe_seed',
        'n_train_events', 'n_train_particles', 'n_eval_events', 'n_eval_particles', 'n_test_particles',
        'n_probe_params', 'steps', 'best_step', 'read_z1', 'kn_z1', 'delta_z1', 'read_z1_min',
        'step_z1_min', 'read_z1_test', 'kn_z1_test', 'delta_z1_test',
        'sec_encoder', 'sec_probe', 'gpu_name', 'host', 'timestamp']

if args.untrained is not None:
    arm, setting, seed, run_dir = 'untrained', 'untrained', args.untrained, ''
    ckpt_tag, enc_key = 'random-init', 'random_init(no_checkpoint)'
    enc = C.untrained_encoder(seed)
else:
    tags = sorted({r['checkpoint_tag'] for r in R.runs() if r['setting'] == args.setting
                   and r['seed'] == str(args.seed)}, key=lambda t: int(t.split('-ep')[1]))
    row = R.find(setting=args.setting, seed=args.seed, checkpoint_tag=args.tag or tags[-1])
    arm, setting, seed, run_dir, ckpt_tag = row['arm'], row['setting'], args.seed, row['run_folder'], row['checkpoint_tag']
    enc, enc_key = R.load(row, args.root, prefer=args.key)

t0 = time.time()
tr, ev = C.get_events(cache=args.cache)
te = C.get_test_events(cache=args.cache)
kn, kn_te = C.know_nothing(ev), C.know_nothing(te)
tok_tr, tok_ev, tok_te = (C.extract_tokens(enc, d['imgs'], chunk=256) for d in (tr, ev, te))
del enc
torch.cuda.empty_cache()
sec_enc = time.time() - t0

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tag = f'{arm.replace(" ", "-")}_s{seed}_st{steps}'
packed = dict(arm=arm, setting=setting, train_seed=seed, run_dir=run_dir, ckpt_tag=ckpt_tag,
              encoder_key=enc_key, steps=steps, know_nothing_z1=np.float32(kn['z1']),
              know_nothing_z1_test=np.float32(kn_te['z1']), test_seed_base=np.int64(C.TEST_SEED_BASE))
weights = {}
csv_path = out / 'fits_z1.csv'
for ps in args.probe_seeds:
    t1 = time.time()
    fit = C.fit_z1(tok_tr, tr, tok_ev, ev, tok_te, te, probe_seed=ps, steps=steps)
    packed[f'pred_ps{ps}'] = fit['pred']
    packed[f'pred_test_ps{ps}'] = fit['pred_test']
    packed[f'curve_ps{ps}'] = fit['curve']
    packed[f'best_step_ps{ps}'] = np.int32(fit['step'])
    packed['truth'] = fit['truth']
    packed['truth_test'] = fit['truth_test']
    weights[f'z1big_ps{ps}'] = fit['state']
    row = dict(arm=arm, setting=setting, train_seed=seed, run_dir=run_dir, checkpoint_tag=ckpt_tag,
               encoder_key=enc_key, probe='z1big', probe_seed=ps,
               n_train_events=tr['imgs'].shape[0], n_train_particles=int(tr['mask'].sum()),
               n_eval_events=ev['imgs'].shape[0], n_eval_particles=int(ev['mask'].sum()),
               n_test_particles=int(te['mask'].sum()), n_probe_params=fit['n_params'],
               steps=steps, best_step=fit['step'],
               read_z1=round(fit['med'], 5), kn_z1=round(kn['z1'], 5),
               delta_z1=round(kn['z1'] - fit['med'], 5),
               read_z1_min=round(fit['min'], 5), step_z1_min=fit['argmin_step'],
               read_z1_test=round(fit['med_test'], 5), kn_z1_test=round(kn_te['z1'], 5),
               delta_z1_test=round(kn_te['z1'] - fit['med_test'], 5),
               sec_encoder=round(sec_enc, 1), sec_probe=round(time.time() - t1, 1),
               gpu_name=C.gpu_name(), host=socket.gethostname(),
               timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))
    new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f'  z1big ps{ps}  z1 sel {fit["med"]:.4f} test {fit["med_test"]:.4f} '
          f'd_test {row["delta_z1_test"]:+.4f} @step {fit["step"]}', flush=True)
np.savez_compressed(out / f'{tag}.npz', **packed)
torch.save(weights, out / f'{tag}_probe.pt')
