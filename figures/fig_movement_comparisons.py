"""Figure: movement and shape readability across three configuration comparisons
(predictor capacity, training objective, training condition).

One point per training seed (readability: main probe, test set, two probe draws
averaged; movement: per-token median at N = 1000), group medians, and the
untrained encoders' median movement as a reference line in panel (a). The
deeper predictor is summarized by its two seed groups, never by its pooled
median in the gap between them. Collapsed BYOL* encoders are left out.
Reads results/readability.csv and results/movement.csv.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import data as D

UNTRAINED_MOVE = D.untrained_movement()
raw = D.movement_comparison_table()
data = raw[raw.collapsed.eq(0)].copy()
purple, teal, rust = '#754C91', '#287E91', '#B45D3D'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': .65, 'xtick.major.width': .65,
    'ytick.major.width': .65, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white',
})

groups = {
    'default': ('shared_default', None),
    'deeper_low': ('shared_deeper', [1, 3, 5]),
    'deeper_high': ('shared_deeper', [4, 6, 7]),
    'wider': ('shared_wider', None),
    'simclr': ('shared_simclr', None),
    'byol': ('shared_byol', None),
    'constant': ('constant_default', None),
}
summary = {}
for name, (cohort, seeds) in groups.items():
    d = data[data.cohort.eq(cohort)]
    if seeds is not None:
        d = d[d.train_seed.isin(seeds)]
    summary[name] = {
        'cohort': cohort, 'seeds': sorted(d.train_seed.tolist()),
        'n': len(d), 'movement': d.movement.median(),
        'readability': d.readability.median(),
    }

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.85))
fig.subplots_adjust(left=.077, right=.98, bottom=.275, top=.965, wspace=.27)
for i, ax in enumerate(axes):
    ax.set_ylim(-.012, .325)
    ax.set_yticks([0, .1, .2])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x:.1f}'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#777777')
    ax.tick_params(colors='#333333', pad=3)
    ax.grid(axis='y', color='#E7E7E7', linewidth=.5, zorder=0)
    ax.set_xlabel('Movement', labelpad=4)
    if i == 0:
        ax.set_xlim(0, .26)
        ax.set_xticks([0, .1, .2])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x:.1f}'))
        ax.set_ylabel('Shape readability', labelpad=5)
    elif i == 1:
        ax.set_xlim(0, .047)
        ax.set_xticks([0, .02, .04])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x:.2f}'))
        ax.tick_params(labelleft=False)
    else:
        ax.set_xlim(0, .10)
        ax.set_xticks([0, .05, .10])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x:.2f}'))
        ax.tick_params(labelleft=False)

def point(ax, key, color=purple, marker='o', hollow=False):
    s = summary[key]
    d = data[data.cohort.eq(s['cohort']) & data.train_seed.isin(s['seeds'])]
    ax.scatter(d.movement, d.readability, s=10,
               marker=marker, facecolor='white', edgecolor=color,
               linewidth=.65, alpha=.60, zorder=3)
    ax.scatter(s['movement'], s['readability'], s=32,
               marker=marker, facecolor=color,
               edgecolor='white', linewidth=.55, alpha=.85, zorder=4)

def label(ax, x, y, text, color=purple, ha='left', va='center'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=7, color=color,
            linespacing=1.25, zorder=5)

a, b, c = axes
for key in ['default', 'wider']:
    point(a, key)
for key in ['deeper_low', 'deeper_high']:
    point(a, key, marker='s', hollow=True)
label(a, .027, summary['default']['readability'], 'Default I-JEPA*')
label(a, .045, summary['deeper_low']['readability'], 'Deeper,\nlower three')
label(a, .156, .226, 'Deeper,\nupper three', ha='right')
label(a, .225, .207, 'Wider', ha='right')   # kept clear of the untrained line
a.axvline(UNTRAINED_MOVE, color='#555555', lw=.65, ls=(0, (4, 2.5)), zorder=1)
label(a, UNTRAINED_MOVE - .004, .004, 'Untrained', '#555555', ha='right', va='bottom')

point(b, 'default')
point(b, 'simclr', teal)
point(b, 'byol', rust)
label(b, .018, .058, 'Default I-JEPA*')
label(b, summary['simclr']['movement'], .215, 'SimCLR*', teal, ha='center')
label(b, summary['byol']['movement'], .177, 'BYOL*', rust, ha='center')

point(c, 'default')
point(c, 'constant', marker='s')
label(c, .002, .092, 'Shared\n(default)')
label(c, .038, .080, 'Constant')

def message(ax, text):
    ax.text(.035, .975, text, transform=ax.transAxes, ha='left', va='top', fontsize=7,
            color='#333333', linespacing=1.3, zorder=6,
            bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.25', fc='#EFEFEF', ec='none'))

message(a, 'Readability rises\nwith movement')
message(b, 'Movement does not\ntrack readability')
message(c, 'Same readability,\ndifferent movement')

for ax, text in zip(axes, ['(a) Predictor capacity', '(b) Training objective', '(c) Training condition']):
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, .056, text, ha='center', fontsize=7)

fig.savefig(OUT / 'movement_comparisons.pdf')
fig.savefig(OUT / 'movement_comparisons.png', dpi=260)
plt.close(fig)
print('wrote', OUT / 'movement_comparisons.pdf')
