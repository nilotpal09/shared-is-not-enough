"""Score saved probes on the independent test set (no refitting).

For every probe file in a folder: rebuild the encoder it was fitted on (from the
released checkpoint, or from its seed for the untrained reference), extract
tokens for the selection set and the test set, load the saved probe weights at
the selected step, and record the median |error| per target on both sets, plus
the value recomputed from the probe file's own stored predictions as a loading
check.
    delta_<t>_test = know_nothing_test(t) - MedAE_test(t)     <- the paper's score
    delta_<t>_sel  = know_nothing_sel(t)  - MedAE_sel(t)      <- the fit's own score

Folders in the Zenodo record (under <root>/probes/), with --kind and --probe-n:
    npz_c40k/N<n>/ps<k>                       big, n = 30 ... 10000   main probe
    npz_c40k/N1000_online/ps<k>               big                     BYOL* online copy
    c40k_online_masked/npz_c40k/N1000_online/ps<k>   big              I-JEPA* online copy
    c40k_simple/npz_c40k/N1000/ps<k>          simple                  small probe
    c40k_simple_cap/npz_c40k/N1000/ps<k>      simple                  small probe, deeper/wider
    c40k_traj/ep<t>/npz_c40k/N1000/ps<k>      big                     training over time
    c40k_headwidth/npz_c40k/N1000/ps<k>       big                     narrower projectors
    split/npz_z1single, split_capcheck/npz_z1single   z1big           single-target probe

    python probes/score_test.py --tree npz_c40k/N1000/ps1 --probe-n 1000 --kind big
    python probes/score_test.py --tree split/npz_z1single --kind z1big
"""
import argparse
import csv
import socket
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--tree', required=True, help='probe folder: relative to <root>/probes, or any folder of fit_probes.py output')
ap.add_argument('--probe-n', type=int, default=1000, help='probe graininess the probes were fitted at')
ap.add_argument('--kind', default='big', choices=['big', 'simple', 'z1big'])
ap.add_argument('--root', default=None, help='unzipped Zenodo record (default ./checkpoints)')
ap.add_argument('--cache', default=None, help='probe-set cache (default ./cache)')
ap.add_argument('--out', default=None, help='output CSV (default results_rescored/<tree id>.csv)')
ap.add_argument('--limit', type=int, default=0, help='first k probe files only')
args = ap.parse_args()

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import common as C                                   # noqa: E402
import runs as R                                     # noqa: E402

probes_root = C.data_root(args.root) / 'probes'
tree = Path(args.tree) if Path(args.tree).is_absolute() else probes_root / args.tree
assert tree.is_dir(), tree
tree_id = (str(tree.relative_to(probes_root)) if tree.is_relative_to(probes_root) else tree.name).replace('/', '_')
out_csv = Path(args.out) if args.out else C.REPO / 'results_rescored' / f'{tree_id}.csv'
out_csv.parent.mkdir(parents=True, exist_ok=True)
n = args.probe_n
targets = ('z1',) if args.kind == 'z1big' else C.TARGETS

COLS = ['tree', 'arm', 'train_seed', 'run_dir', 'checkpoint_tag', 'encoder_key',
        'probe', 'probe_seed', 'probe_n', 'n_probe_params', 'best_step',
        'n_sel_particles', 'n_test_particles']
for t in targets:
    COLS += [f'read_{t}_npz', f'read_{t}_sel', f'kn_{t}_sel', f'delta_{t}_sel',
             f'read_{t}_test', f'kn_{t}_test', f'delta_{t}_test']
COLS += ['gpu_name', 'host', 'timestamp']


def append_row(path, row):
    new = not path.exists()
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, '') for c in COLS})


def load_encoder(arm, seed, run_dir, ckpt_tag, encoder_key):
    if arm == 'untrained':
        return C.untrained_encoder(seed)
    row = R.find(run_folder=Path(run_dir).name, checkpoint_tag=ckpt_tag)
    prefer = 'encoder' if encoder_key == 'encoder' else 'target_encoder'
    enc, key = R.load(row, args.root, prefer=prefer)
    assert key == encoder_key, (key, encoder_key)
    return enc


def medae(p, t):
    return (p - t).abs().median(dim=0).values.cpu().numpy()


# ---- the two held-out sets ---------------------------------------------------
t0 = time.time()
_, ev = C.get_events(n, cache=args.cache)
te = C.get_test_events(n, cache=args.cache)
kn_sel, kn_te = C.know_nothing(ev), C.know_nothing(te)
qkind = 'big' if args.kind == 'z1big' else args.kind
sel = C._prep(ev, None, C.DEVICE, C.HALF_W, C.HALF_H, qkind)
tst = C._prep(te, None, C.DEVICE, C.HALF_W, C.HALF_H, qkind)
if args.kind == 'z1big':
    sel['t3'], tst['t3'] = sel['t3'][..., :1], tst['t3'][..., :1]
print(f'score_test | {tree_id} | N={n} | {args.kind} | sel {int(ev["mask"].sum())} '
      f'test {int(te["mask"].sum())} particles | {C.gpu_name()} | {time.time()-t0:.0f}s', flush=True)

files = sorted(tree.glob('*.npz'))[:args.limit or None]
n_done = 0
for npz in files:
    tag = npz.stem
    pt = tree / f'{tag}_probe.pt'
    if not pt.exists():
        print(f'  {tag}: no probe weights, skipped', flush=True)
        continue
    d = np.load(npz, allow_pickle=True)
    arm, seed = str(d['arm']), int(d['train_seed'])
    run_dir, ckpt_tag, enc_key = str(d['run_dir']), str(d['ckpt_tag']), str(d['encoder_key'])
    weights = torch.load(pt, map_location='cpu')
    keys = [w for w in weights if w.startswith(args.kind + '_ps')]
    t1 = time.time()
    enc = load_encoder(arm, seed, run_dir, ckpt_tag, enc_key)
    with torch.no_grad():
        sel['tok'] = C.extract_tokens(enc, ev['imgs'], chunk=128)
        tst['tok'] = C.extract_tokens(enc, te['imgs'], chunk=128)
    del enc
    torch.cuda.empty_cache()
    for w in keys:
        ps = w.split('_ps')[1]
        model = C.build_probe(args.kind, token_dim=sel['tok'].shape[-1]).to(C.DEVICE)
        model.load_state_dict(weights[w])
        med_sel = medae(*C._eval_preds(model, sel))
        med_te = medae(*C._eval_preds(model, tst))
        if args.kind == 'z1big':
            med_npz = np.median(np.abs(d[f'pred_ps{ps}'] - d['truth']))[None]
            best = int(d[f'best_step_ps{ps}'])
        else:
            med_npz = np.median(np.abs(d[f'pred_{args.kind}_ps{ps}'] - d['truth']), axis=0)
            best = int(d[f'best_step_{args.kind}_ps{ps}'])
        row = dict(tree=tree_id, arm=arm, train_seed=seed, run_dir=Path(run_dir).name,
                   checkpoint_tag=ckpt_tag, encoder_key=enc_key, probe=args.kind,
                   probe_seed=int(ps), probe_n=n,
                   n_probe_params=sum(q.numel() for q in model.parameters()),
                   best_step=best, n_sel_particles=int(ev['mask'].sum()),
                   n_test_particles=int(te['mask'].sum()),
                   gpu_name=C.gpu_name(), host=socket.gethostname(),
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'))
        for i, t in enumerate(targets):
            row[f'read_{t}_npz'] = round(float(med_npz[i]), 5)
            row[f'read_{t}_sel'] = round(float(med_sel[i]), 5)
            row[f'kn_{t}_sel'] = round(kn_sel[t], 5)
            row[f'delta_{t}_sel'] = round(kn_sel[t] - float(med_sel[i]), 5)
            row[f'read_{t}_test'] = round(float(med_te[i]), 5)
            row[f'kn_{t}_test'] = round(kn_te[t], 5)
            row[f'delta_{t}_test'] = round(kn_te[t] - float(med_te[i]), 5)
        append_row(out_csv, row)
        n_done += 1
        print(f'  {arm:<20s} s{seed} {ckpt_tag:<11s} ps{ps}  z1: sel {med_sel[0]:.4f} '
              f'test {med_te[0]:.4f}  d_test {row["delta_z1_test"]:+.4f}  [{time.time()-t1:.0f}s]', flush=True)
print(f'DONE {tree_id}: {n_done} rows -> {out_csv} in {(time.time()-t0)/60:.1f} min', flush=True)
