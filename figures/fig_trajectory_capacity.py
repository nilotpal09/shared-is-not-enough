"""Figure: the shared factor over training, three I-JEPA* predictors beside
SimCLR*: readability against training epoch on a linear axis.

Main probe (40,000 steps, test set) on checkpoints at epochs 5-400 (training
over time) and the final checkpoints; six seeds per setting, the two probe
draws averaged per seed. Reads results/readability.csv; writes one PNG.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S
import data as D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / 'trajectory_capacity.png'
W1, H1 = 6.4, 5.0

ARMS = {
    'ijepa_shared_default': (S.MUTED,   'o', 'I-JEPA*, default predictor'),
    'ijepa_shared_deeper':  ('#1f4e9c', 's', 'deeper predictor'),
    'ijepa_shared_wider':   ('#6fb3f2', '^', 'wider predictor'),
    'simclr_shared':        (S.ORANGE,  'o', 'SimCLR*'),
}
# untrained reference: median over the eight random inits, to three decimals (0.310)
UNT_READ = round(float(D.per_seed(D.readability(), 'untrained').median()), 3)


def load():
    """{(setting, seed): {epoch: (d(z1),)}}, the two probe draws averaged."""
    rows = [D.readability('training over time'), D.readability('main')]
    out = {}
    for d in rows:
        for r in d[d.setting.isin(list(ARMS))].itertuples():
            ep = int(r.checkpoint_tag.split('-ep')[1])
            out.setdefault((r.setting, int(r.train_seed)), {}).setdefault(ep, []).append(float(r.delta_z1_test))
    return {k: {e: (float(np.mean(v)),) for e, v in eps.items()} for k, eps in out.items()}


def main():
    S.apply(fig_w=11.2)
    D = load()
    fig, axA = plt.subplots(1, 1, figsize=(W1, H1))

    # Each reference label is placed at an explicit height in clear space, not
    # hugging its rule: in (b) the strip either side of 0.23 is occupied at every
    # epoch (the width-96 median runs along it from epoch 300, seed lines spike
    # above it before epoch 100), so (b) gets headroom and the label sits above
    # everything drawn.
    panels = ((axA, 0, 'readability of the factor  $d(z_1)$', UNT_READ,
               f'untrained readability {UNT_READ:.3f}', (-0.012, 0.352), 0.323),)
    med_report = {}
    for ax, i, ylab, ref, reflab, ylim, reftexty in panels:
        ax.axhline(ref, color=S.INK2, lw=S.f(1.2), ls=(0, (5, 3)), zorder=1)
        for arm, (col, mk, lab) in ARMS.items():
            seeds = sorted(s for (a, s) in D if a == arm)
            eps = sorted({e for s in seeds for e in D[(arm, s)]})
            for s in seeds:
                e = sorted(D[(arm, s)])
                ax.plot(e, [D[(arm, s)][x][i] for x in e], color=col,
                        lw=S.f(0.7), alpha=0.30, zorder=2)
            med = [np.median([D[(arm, s)][e][i] for s in seeds if e in D[(arm, s)]])
                   for e in eps]
            ax.plot(eps, med, color=col, lw=S.f(2.2), zorder=4)
            ax.plot(eps, med, ls='none', marker=mk, markersize=S.f(5.4),
                    markerfacecolor='none', markeredgecolor=col,
                    markeredgewidth=S.f(1.5), zorder=5)
            med_report[(arm, i)] = dict(zip(eps, med))
        ax.set_xlim(-12, 512)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.set_ylabel(ylab)
        ax.set_xlabel('training epoch')
        S.tidy(ax)
        ax.text(0.985, reftexty, reflab, transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', color=S.INK2, fontsize=S.f(8.8))

    handles = [Line2D([], [], color=col, lw=S.f(2.2), marker=mk,
                      markerfacecolor='none', markeredgecolor=col,
                      markeredgewidth=S.f(1.5), markersize=S.f(6), label=lab)
               for col, mk, lab in ARMS.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, S.LEGEND_Y),
               ncol=2, frameon=False, columnspacing=2.0, handletextpad=0.6)

    fig.subplots_adjust(left=0.125, right=0.975, top=0.835, bottom=0.115)
    fig.savefig(OUT, dpi=220, facecolor=S.SURFACE)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
