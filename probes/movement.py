"""Movement of encoders under shape-only resampling (per token, no probe).

    python probes/movement.py --setting ijepa_shared_default --seeds 1 3 4 5 6 7
    python probes/movement.py --setting ijepa_shared_default --seeds 1 --tag jepa-ep100
    python probes/movement.py --untrained 1 2 3

M = 64 truths x K = 8 replicas at the probe graininess (default N = 1000):
spots pinned (fixed_spot_seed 777), alpha/T redrawn per replica. Per token
position, r = intra / inter (cosine distances between L2-normalised tokens);
reported as the median and quartiles over the 256 tokens, plus the value for
mean-pooled tokens. GPU only: the replicas come from the device's random stream.
"""
import argparse
import csv
import socket
import time
from pathlib import Path

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--setting', help='setting name (probes/runs.csv)')
g.add_argument('--untrained', type=int, nargs='+', metavar='SEED', help='random-init encoders')
ap.add_argument('--seeds', type=int, nargs='+', help='training seeds (with --setting)')
ap.add_argument('--tag', default=None, help='checkpoint tag (default: the final one, ep500)')
ap.add_argument('--key', default=None, choices=['target_encoder', 'encoder'])
ap.add_argument('--probe-n', type=int, default=1000)
ap.add_argument('--root', default=None, help='unzipped Zenodo record (default ./checkpoints)')
ap.add_argument('--out', default='movement.csv')
args = ap.parse_args()

import torch                                         # noqa: E402

import common as C                                   # noqa: E402
import runs as R                                     # noqa: E402

assert C.DEVICE.type == 'cuda', 'movement is measured on the GPU'
COLS = ['arm', 'setting', 'train_seed', 'run_dir', 'checkpoint_tag', 'encoder_key', 'probe_n',
        'r_token_med', 'r_token_q1', 'r_token_q3', 'r_pooled', 'gpu_name', 'host', 'timestamp']
n = args.probe_n
imgs = C.resim_images(C.make_resim_collator(n), n)
torch.cuda.empty_cache()

jobs = []
if args.untrained:
    jobs = [('untrained', 'untrained', s, '', 'random-init', 'random_init(no_checkpoint)', None) for s in args.untrained]
else:
    assert args.seeds, '--seeds is required with --setting'
    for s in args.seeds:
        tags = sorted({r['checkpoint_tag'] for r in R.runs() if r['setting'] == args.setting
                       and r['seed'] == str(s)}, key=lambda t: int(t.split('-ep')[1]))
        row = R.find(setting=args.setting, seed=s, checkpoint_tag=args.tag or tags[-1])
        jobs.append((row['arm'], row['setting'], s, row['run_folder'], row['checkpoint_tag'], None, row))

out = Path(args.out)
new = not out.exists()
with open(out, 'a', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()
    for arm, setting, seed, run_dir, tag, key, row in jobs:
        if row is None:
            enc = C.untrained_encoder(seed)
        else:
            enc, key = R.load(row, args.root, prefer=args.key)
        r, pooled, _ = C.per_token_resim(enc, imgs)
        del enc
        torch.cuda.empty_cache()
        med, q1, q3 = C.r_summary(r)
        w.writerow(dict(arm=arm, setting=setting, train_seed=seed, run_dir=run_dir, checkpoint_tag=tag,
                        encoder_key=key, probe_n=n, r_token_med=round(med, 6), r_token_q1=round(q1, 6),
                        r_token_q3=round(q3, 6), r_pooled=round(pooled, 6), gpu_name=C.gpu_name(),
                        host=socket.gethostname(), timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
        fh.flush()
        print(f'{setting:<34s} s{seed} {tag:<11s} r_token_med {med:.6f}  pooled {pooled:.6f}', flush=True)
