"""Simulator facts behind statements about the images:

(a) shower-maximum depth: per-particle T (the depth of the longitudinal maximum,
    in simulator depth units) from the stage-1 draw of the training generator,
    its median, 5% and 95% quantiles and the share per depth layer;
(b) cell energies at N = 100, 1000, 10^4: the largest cell, the median of the
    per-event largest cell, and x = log(1+E)/log(1+5500) over occupied cells
    (E > 0), the model input's range;
(c) the non-empty cell counts of the graininess figure's event
    (example_event.py: depth-summed cells > 0), N = 100, 1000, 10^4.

Writes simulator_facts.csv (fact, n_spots, statistic, value). The cell-energy
values (b) vary in the sixth significant digit between GPU runs (atomic
additions in the simulator); everything else is exact.

    python probes/simulator_facts.py
"""
import argparse
import csv
import math

import numpy as np
import torch

import common as C
import example_event as X

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='simulator_facts.csv')
OUT = ap.parse_args().out
DEVICE = C.DEVICE
N_EVENTS, BATCH, SEED0 = 2000, 50, 20260924
rows = []


def collator(calo_cfg, gen_cfg, n_spots):
    return X.make_collator(n_spots)


cfg = C.base_cfg()
calo, gen = cfg['data']['calo_cfg'], cfg['data']['generator_cfg']
scale = float(cfg['data']['norm']['scale'])
layer_depth = float(calo['depth']) / int(calo['N_cells_z'])
print(f'device {DEVICE} | layer depth {layer_depth} | norm scale {scale}', flush=True)

# (a) shower-maximum depth
bc = collator(calo, gen, 1000)
T_all = []
for b in range(N_EVENTS // BATCH):
    s = SEED0 + b
    p_E, p_x, p_y = bc._generate_truth(batch_size=BATCH, seed=s)
    lp = bc._compute_long_params(p_E, bc._sim_seed(s, 0))
    T_all.append(lp['T'].detach().cpu().double().numpy().ravel())
T = np.concatenate(T_all)
T = T[np.isfinite(T)]
layer = np.minimum(np.floor(T / layer_depth).astype(int) + 1, int(calo['N_cells_z']))
rows += [('shower_max_depth', 1000, 'n_particles', len(T)),
         ('shower_max_depth', 1000, 'median_T', float(np.median(T))),
         ('shower_max_depth', 1000, 'q05_T', float(np.quantile(T, 0.05))),
         ('shower_max_depth', 1000, 'q95_T', float(np.quantile(T, 0.95))),
         ('shower_max_depth', 1000, 'layer_depth', layer_depth)]
rows += [('shower_max_depth', 1000, f'share_layer{k}', float((layer == k).mean()))
         for k in range(1, int(calo['N_cells_z']) + 1)]
print('(a) median T', np.median(T), 'layer shares', [(k, round(float((layer == k).mean()), 3)) for k in range(1, 5)], flush=True)

# (b) cell energies and normalized occupied-cell range
denom = math.log1p(scale)
for n in (100, 1000, 10000):
    bc = collator(calo, gen, n)
    emax, xs = [], []
    for b in range(400 // BATCH):
        s = SEED0 + 1000 + b
        p_E, p_x, p_y = bc._generate_truth(batch_size=BATCH, seed=s)
        grid = bc._simulate_once(p_E, p_x, p_y, bc._sim_seed(s, 0)).detach().double().cpu()
        emax.append(grid.flatten(1).max(dim=1).values.numpy())
        occ = grid[grid > 0]
        xs.append((torch.log1p(occ) / denom).numpy())
    emax, x = np.concatenate(emax), np.concatenate(xs)
    rows += [('cell_energy', n, 'n_events', len(emax)),
             ('cell_energy', n, 'max_cell_energy_overall', float(emax.max())),
             ('cell_energy', n, 'median_of_per_event_max', float(np.median(emax))),
             ('cell_energy', n, 'occupied_x_min', float(x.min())),
             ('cell_energy', n, 'occupied_x_q001', float(np.quantile(x, 0.001))),
             ('cell_energy', n, 'occupied_x_q01', float(np.quantile(x, 0.01))),
             ('cell_energy', n, 'occupied_x_median', float(np.median(x))),
             ('cell_energy', n, 'occupied_x_max', float(x.max())),
             ('cell_energy', n, 'share_occupied_x_above_1', float((x > 1).mean()))]
    print(f'(b) N={n}: max cell E {emax.max():.1f}, median per-event max {np.median(emax):.1f}, '
          f'occupied x min {x.min():.3f} q0.1% {np.quantile(x, 0.001):.3f} q1% {np.quantile(x, 0.01):.3f} '
          f'max {x.max():.3f}', flush=True)

# (c) Fig 7 lit-cell counts, the figure's own event
seed = X.find_centered_jet_seed()
for N, img in X.simulate_at_n_spots(seed, n_spots_list=(100, 1000, 10000)):
    rows.append(('fig7_lit_cells', N, f'count_seed{seed}', int((img > 0).sum())))
    print(f'(c) N={N}: {int((img > 0).sum())} cells lit (seed {seed}; printed 1063/2884/4068)', flush=True)

with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['fact', 'n_spots', 'statistic', 'value'])
    w.writerows(rows)
print(f'wrote {OUT} ({len(rows)} rows)')
