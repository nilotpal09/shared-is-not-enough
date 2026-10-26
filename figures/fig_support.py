"""Figure: readability by sharing condition (a) and by training graininess (b).

Raw test-set d(z1), main probe, 40,000 steps, final encoders; two probe fits
averaged per training seed. Collapsed BYOL* encoders (shared seeds 4 and 7,
constant seed 6) left out. Reads results/readability.csv.

Physical canvas 6.0 x 2.5 inches; include at its own width for 7-pt text.
"""

from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import _style as S
from data import seeds

PLUM, PETROL, RUST = '#754C91', '#287E91', '#B45D3D'
COLOURS = {'I-JEPA*': PLUM, 'SimCLR*': PETROL, 'BYOL*': RUST}
METHODS = list(COLOURS)

# Same physical typography as fig_core.py: S.apply(6.5) sets the base to 7 pt.
S.apply(6.5)
fig = plt.figure(figsize=(6.0, 2.5))
left = fig.add_axes([0.09, 0.335, 0.51, 0.55])
right = fig.add_axes([0.657, 0.335, 0.323, 0.55], sharey=left)

left.set_xlim(-0.24, 4.23)
right.set_xlim(-0.48, 3.87)
for ax in (left, right):
    ax.set_ylim(-0.010, 0.26)
    ax.set_yticks([0, 0.1, 0.2])
    S.tidy(ax)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', length=0)
right.tick_params(axis='y', labelleft=False)
left.set_ylabel('readability of shape')

condition_centres = [0.5, 2.05, 3.60]
left.set_xticks(condition_centres)
left.set_xticklabels(['constant', 'shared', 'redrawn'])
diet_positions = [0.0, 1.0, 2.0, 3.38]
right.set_xticks(diet_positions)
right.set_xticklabels(['grainy', 'default', 'smooth', 'mixed'])

for x, sub in zip(condition_centres,
                  ['never\nvaries', 'same in\nboth views', 'differs\nbetween views']):
    left.text(x, -0.18, sub, transform=left.get_xaxis_transform(),
              ha='center', va='top', fontsize=7, color=S.INK2, linespacing=1.05)
# Diet names are defined with N in Section 2.2. Repeating those four values
# under this compact panel crowds the labels without adding a comparison.

# Reuse the freed heading space: move both axes up without changing
# their physical dimensions, leaving room for panel labels underneath.
fig.text(0.345, 0.065, '(a) Shape in the training pairs',
         ha='center', va='bottom', fontsize=7)
fig.text(0.8185, 0.065, '(b) Training graininess',
         ha='center', va='bottom', fontsize=7)
legend = left.legend(handles=[Patch(facecolor=COLOURS[m], linewidth=0, label=m) for m in METHODS],
            loc='lower left', bbox_to_anchor=(0, 1.045), ncol=3,
            borderaxespad=0, handlelength=0.85, handleheight=0.8,
            columnspacing=1.1, handletextpad=0.45, fontsize=7)
fig.canvas.draw()
legend_box = legend.get_window_extent(fig.canvas.get_renderer())
legend_mid_y = (legend_box.y0 + legend_box.y1) / (2 * fig.bbox.height)
fig.text(0.8185, legend_mid_y, 'I-JEPA*, shared pairs',
         ha='center', va='center', fontsize=7, color=S.INK2)
fig.canvas.draw()


def compact_offsets(ax, x, values, diameter_pt=3.1):
    """core_12 placement: smallest horizontal offsets avoiding seed overlap."""
    separation_px = (diameter_pt + 0.6 + 0.25) * fig.dpi / 72
    origin = ax.transData.transform((x, 0))[0]
    scale_x = ax.transData.transform((x + 1, 0))[0] - origin
    placed = []
    offsets = np.zeros(len(values))
    for i in np.argsort(values, kind='stable'):
        yy = ax.transData.transform((x, values[i]))[1]
        candidates = [0.0]
        for ox, oy in placed:
            dy = abs(yy - oy)
            if dy < separation_px:
                dx = np.sqrt(separation_px**2 - dy**2)
                candidates.extend([ox - dx, ox + dx])
        for dx in sorted(candidates, key=lambda value: (abs(value), value)):
            if all((dx - ox)**2 + (yy - oy)**2 >= separation_px**2 - 1e-7
                   for ox, oy in placed):
                offsets[i] = dx / scale_x
                placed.append((dx, yy))
                break
        else:
            raise AssertionError('No non-overlapping seed placement found')
    offsets -= (offsets.min() + offsets.max()) / 2
    assert np.max(np.abs(offsets)) < 0.16
    return offsets


audit = []


def draw_bar(ax, x, key, colour, expected_n, panel, label):
    series = seeds(key).sort_index()
    values = series.to_numpy()
    assert len(values) == expected_n, (key, len(values), expected_n)
    assert np.isfinite(values).all()
    median = float(np.median(values))
    offsets = compact_offsets(ax, x, values)
    # Match core_12's approximately 0.20-inch bars in both panels.
    width_in = 0.20
    ax_width_in = ax.get_position().width * fig.get_figwidth()
    width_data = width_in * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax_width_in
    bars = ax.bar(x, median, width=width_data, color=colour, linewidth=0, zorder=2)
    actual_width_in = bars.patches[0].get_window_extent().width / fig.dpi
    assert abs(actual_width_in - width_in) < 1e-10
    ax.scatter(x + offsets, values, s=3.1**2, facecolors='white',
               edgecolors=colour, linewidths=0.6, zorder=3)
    for seed, score, offset in zip(series.index, values, offsets):
        px, py = ax.transData.transform((x + offset, score))
        radius = 1.85 * fig.dpi / 72
        assert ax.bbox.x0 + radius < px < ax.bbox.x1 - radius
        assert ax.bbox.y0 + radius < py < ax.bbox.y1 - radius
        audit.append(dict(panel=panel, label=label, source_key=key,
                          train_seed=int(seed), d_z1=float(score), median=median))
    print(f'{panel} {label}: n={len(values)}, median={median:.6f}, '
          f'range=[{values.min():.6f}, {values.max():.6f}]')


# Fixed method slots even when no redrawn BYOL was trained.
for centre, condition in zip(condition_centres, ['constant', 'shared', 'redrawn']):
    for offset, method in zip([-0.38, 0.0, 0.38], METHODS):
        if method == 'BYOL*' and condition == 'redrawn':
            continue
        n = (5 if condition == 'constant' else 4) if method == 'BYOL*' else 6
        key = f'{method} {condition}'
        draw_bar(left, centre + offset, key, COLOURS[method], n, 'a', key)

for x, label, key in zip(diet_positions,
                         ['grainy', 'default', 'smooth', 'mixed'],
                         ['ijepa_shared_grainy', 'I-JEPA* shared', 'ijepa_shared_smooth', 'ijepa_shared_mixed']):
    draw_bar(right, x, key, PLUM, 6, 'b', label)

# Check a fact the text states (redrawn above shared at five of six seeds).
redrawn = seeds('I-JEPA* redrawn')
shared = seeds('I-JEPA* shared')
assert redrawn.index.equals(shared.index)
assert int((redrawn > shared).sum()) == 5
assert int(redrawn.idxmin()) == 6

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bbox = fig.get_tightbbox(renderer)
assert bbox.x0 >= 0 and bbox.y0 >= 0
assert bbox.x1 <= 6.0 and bbox.y1 <= 2.5, bbox
fig.savefig(OUT / 'support.png', dpi=300)
fig.savefig(OUT / 'support.pdf')
with (OUT / 'support_data.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
    writer.writeheader()
    writer.writerows(audit)
print('PASS: counts, all-seed bounds, compact seed placement, matched count, canvas bounds, identical 0.20-inch bar widths.')
print(OUT / 'support.png')
