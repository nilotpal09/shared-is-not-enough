"""Figure: readability against probe graininess for the eight training-image settings.

Two panels (shared, single-view). For each training graininess, d(z1) on the
main probe (40,000 steps, test set) at the three probe graininess levels N = 100, 1000, 10^4:
thin per-seed lines and a thick line for the median over seeds. A filled marker
is a probe graininess the encoders were trained on (the mixed setting trained on
the whole range, so its markers are all filled). The untrained reference is the
grey dashed curve over seven probe graininess levels.

Reads results/readability.csv; writes diet_curves.png.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S
import data as D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parent / 'diet_curves.png'
FAM = {'shared': {'coarse': 'ijepa_shared_grainy', 'mixed': 'ijepa_shared_mixed', 'default': 'ijepa_shared_default', 'fine': 'ijepa_shared_smooth'},
       'single-view': {'coarse': 'ijepa_singleview_grainy', 'mixed': 'ijepa_singleview_mixed', 'default': 'ijepa_singleview_default', 'fine': 'ijepa_singleview_smooth'}}
OWN = {'coarse': 100, 'default': 1000, 'fine': 10000}
COL = {'coarse': '#08306b', 'mixed': S.ORANGE, 'default': '#4292c6', 'fine': '#9ecae1'}
LAB = {'coarse': 'grainy, $N=100$', 'mixed': 'mixed, $N\\sim[10^2,10^4]$', 'default': 'default, $N=1000$', 'fine': 'smooth, $N=10^4$'}

class UntrainedKey:
    pass


class HandlerDashCircleDash(HandlerBase):
    """Dashed rule across the handle with the hatched reference circle at its centre."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        y = height / 2 - ydescent
        line = Line2D([-xdescent, width - xdescent], [y, y], color=S.INK2,
                      lw=S.f(1.4), ls=(0, (5, 3)))
        circ = Circle((width / 2 - xdescent, y), S.f(3.4), facecolor='none',
                      edgecolor=S.INK2, lw=S.f(1.3), hatch='////')
        for a in (line, circ):
            a.set_transform(trans)
        return [line, circ]


def main():
    S.apply(fig_w=11.0)
    canon, p100, p1e4 = (D.readability(probe_n=n) for n in (1000, 100, 10000))
    unt = {n: float(D.per_seed(D.readability(probe_n=n), 'untrained').median())
           for n in (30, 100, 200, 500, 1000, 3000, 10000)}
    ux = np.array(sorted(unt)); uy = np.array([unt[n] for n in ux])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True, gridspec_kw={'wspace': 0.0968})
    rungs = np.array([100, 1000, 10000])
    for ax, (fam, arms) in zip(axes, FAM.items()):
        ax.set_xscale('log')
        ax.plot(ux, uy, color=S.INK2, lw=S.f(1.4), ls=(0, (5, 3)), zorder=2)
        ax.scatter(ux, uy, s=S.fs(46), facecolors='none', edgecolors=S.INK2,
                   linewidths=S.f(1.3), hatch='////', zorder=2.5)
        for diet, arm in arms.items():
            ys = np.vstack([D.per_seed(p100, arm).sort_index().values, D.per_seed(canon, arm).sort_index().values, D.per_seed(p1e4, arm).sort_index().values]).T  # seeds x rungs
            for row in ys:
                ax.plot(rungs, row, color=COL[diet], lw=S.f(0.7), alpha=0.35, zorder=3)
            med = np.median(ys, axis=0)
            ax.plot(rungs, med, color=COL[diet], lw=S.f(2.4), zorder=4)
            for n, m in zip(rungs, med):
                own = (OWN.get(diet) == n) or diet == 'mixed'   # filled = inside the training range; mixed trained on the whole range
                ax.scatter([n], [m], s=S.fs(60), marker='o', facecolors=COL[diet] if own else S.SURFACE, edgecolors=COL[diet], linewidths=S.f(1.8), zorder=5)
            print(f'{fam:12s} {diet:8s} median d(z1) at 100/1000/1e4: {med[0]:.3f} / {med[1]:.3f} / {med[2]:.3f}')
        ax.set_xticks([100, 1000, 10000]); ax.set_xticklabels(['100', '1000', r'$10^4$'])
        ax.set_xlim(60, 17000); ax.set_ylim(-0.06, 0.43)
        ax.set_xlabel('probe graininess $N$')
        S.panel_label(ax, 'a' if fam == 'shared' else 'b', fam + ' condition')
        S.tidy(ax)
    axes[0].set_ylabel('readability $d(z_1)$')
    axes[1].text(ux[-1], uy[-1] + 0.012, 'untrained', ha='right', va='bottom', color=S.INK2, fontsize=S.f(9))
    handles = [Line2D([], [], color=COL[d], lw=S.f(2.4), marker='o', markerfacecolor=COL[d], markeredgecolor=COL[d], markersize=S.f(6), label=LAB[d]) for d in ['coarse', 'mixed', 'default', 'fine']]
    labels = [h.get_label() for h in handles] + ['untrained', 'graininess not trained on',
                                                 'one training seed']
    handles += [UntrainedKey(),
                Line2D([], [], ls='none', marker='o', markerfacecolor=S.SURFACE,
                       markeredgecolor=S.INK2, markeredgewidth=S.f(1.4), markersize=S.f(6)),
                Line2D([], [], color=S.INK2, lw=S.f(0.7), alpha=0.5)]
    fig.legend(handles=handles, labels=labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, S.LEGEND_Y), frameon=False, columnspacing=1.8,
               handlelength=2.6, handler_map={UntrainedKey: HandlerDashCircleDash()})
    fig.subplots_adjust(top=0.82, bottom=0.235, left=0.08, right=0.98, wspace=0.0968)
    S.report(fig, 'diet', 11.0)
    fig.savefig(OUT, dpi=220); print('wrote', OUT)

if __name__ == '__main__':
    main()
