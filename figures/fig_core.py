"""Figure: readability of the shared factor at the end of training for SimCLR*,
BYOL* and I-JEPA* with the default, deeper and wider predictors, all on the
same shared pairs. Bars: medians; circles: training seeds (the two probe draws
averaged). Collapsed BYOL* encoders are left out. Reads results/readability.csv.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
from pathlib import Path
import sys
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import _style as S
from data import seeds
PLUM, PETROL, RUST = '#754C91', '#287E91', '#B45D3D'
S.apply(6.5)
W, H = 4.7, 2.3
fig = plt.figure(figsize=(W, H))

# ---------------- objective and predictor comparisons ----------------
ax = fig.add_axes([0.145, 0.23, 0.835, 0.62])
spec = [(0.20, 'SimCLR*', 'SimCLR* shared', PETROL), (0.92, 'BYOL*', 'BYOL* shared', RUST),
        (1.64, 'I-JEPA*\ndefault predictor', 'I-JEPA* shared', PLUM),
        (3.55, 'deeper', 'I-JEPA* deeper', PLUM), (4.25, 'wider', 'I-JEPA* wider', PLUM)]
ax.set_xlim(-0.6, 5.1)
# Leave room for the whole marker at the slightly negative default seed.
ax.set_ylim(-0.010, 0.26)
ax.set_yticks([0, 0.1, 0.2])
ax.set_xticks([x for x, *_ in spec])
ax.set_xticklabels([lab for _, lab, *_ in spec], linespacing=1.0)
ax.set_ylabel('readability of the shared factor')
S.tidy(ax)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.canvas.draw()


def point_offsets(x, values, diameter_pt=3.1):
    """Compact collision-free placement, calculated at the final physical size."""
    spacing_px = (diameter_pt + 0.6 + 0.25) * fig.dpi / 72
    origin = ax.transData.transform((x, 0))[0]
    scale_x = ax.transData.transform((x + 1, 0))[0] - origin
    placed = []
    result = np.zeros(len(values))
    for i in np.argsort(values, kind='stable'):
        yy = ax.transData.transform((x, values[i]))[1]
        candidates = [0.0]
        # Each nearby marker supplies the two closest horizontal positions
        # that avoid it. This packs more tightly than fixed jitter columns.
        for ox, oy in placed:
            dy = abs(yy - oy)
            if dy < spacing_px:
                dx = np.sqrt(spacing_px**2 - dy**2)
                candidates.extend([ox - dx, ox + dx])
        for dx in sorted(candidates, key=lambda a: (abs(a), a)):
            if all((dx - ox)**2 + (yy - oy)**2 >= spacing_px**2 - 1e-7
                   for ox, oy in placed):
                result[i] = dx / scale_x
                placed.append((dx, yy))
                break
        else:
            raise AssertionError('No non-overlapping marker placement found')
    result -= (result.min() + result.max()) / 2
    assert np.max(np.abs(result)) < 0.16
    return result


for x, lab, name, color in spec:
    v = seeds(name).values
    median = float(np.median(v))
    offsets = point_offsets(x, v)
    ax.bar(x, median, width=0.29, color=color, linewidth=0, zorder=2)
    ax.scatter(x + offsets, v, s=3.1**2, facecolors='white',
               edgecolors=color, linewidths=0.6, zorder=3)
    print(f'{name}: full horizontal seed spread={np.ptp(offsets):.4f} category units')
    print(f'{name}: seeds={len(v)}, median={median:.7f}, '
          f'range=[{v.min():.7f}, {v.max():.7f}]')

tr = mt.blended_transform_factory(ax.transData, ax.transAxes)
ax.plot([2.60, 2.60], [0, 1.16], transform=tr, color=S.GRID,
        lw=S.f(0.8), clip_on=False, zorder=1)
ax.text(0.92, 1.06, 'Training objective', transform=tr, ha='center', va='bottom',
        fontsize=S.f(9.5), color=S.INK)
ax.text(3.90, 1.06, 'I-JEPA* predictor', transform=tr, ha='center', va='bottom',
        fontsize=S.f(9.5), color=S.INK)

fig.savefig(OUT / 'core.png', dpi=300)
fig.savefig(OUT / 'core.pdf')
print(OUT / 'core.png')
