"""Shared plot style for the paper's figures.

Palette: a validated light-mode categorical palette, used
by ENTITY, never by value:

    slot 1  blue    #2a78d6
    slot 2  orange  #eb6834
    slot 3  aqua    #1baf7a

Scatter/dot forms carry a three-slot cap under the all-pairs validation, so no
figure here uses more than three hues. Where more than three things need
telling apart, they are separated by position and direct labels instead.

Everything renders on white, because these go into a TMLR PDF. No dark mode.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE   = '#2a78d6'   # series 1
ORANGE = '#eb6834'   # series 2
AQUA   = '#1baf7a'   # series 3

INK      = '#0b0b0b'   # text primary
INK2     = '#52514e'   # text secondary
MUTED    = '#8b8a86'   # de-emphasised marks
GRID     = '#e3e2df'   # hairline grid, one shade off the surface
SURFACE  = '#ffffff'

TEXTWIDTH_IN = 6.5      # TMLR \textwidth; every figure is included at width=\textwidth
BASE_PT      = 9.5      # the base the scripts were written against
TARGET_PT    = 7.0      # rendered base font we want on the page, for every figure
F = 1.0                 # figure-space multiplier set by apply(fig_w)

def apply(fig_w=None):
    """Call with the figure width in inches. Sizes written in the scripts (fontsize,
    s, lw, markersize) are multiplied by F so that, after \includegraphics scales
    the figure to TEXTWIDTH_IN, a size written as 9.5 renders at TARGET_PT and
    every figure's text and markers come out the same size on the page."""
    global F
    if fig_w is not None:
        F = (TARGET_PT / BASE_PT) * (fig_w / TEXTWIDTH_IN)
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.size': 9.5 * F,
        'axes.labelsize': 10 * F,
        'axes.titlesize': 11 * F,
        'axes.labelcolor': INK,
        'axes.edgecolor': GRID,
        'axes.linewidth': 0.8 * F,
        'xtick.color': INK2, 'ytick.color': INK2,
        'xtick.labelsize': 9.5 * F, 'ytick.labelsize': 9.5 * F,
        'xtick.major.size': 0, 'ytick.major.size': 3 * F,
        'xtick.major.width': 0.8 * F, 'ytick.major.width': 0.8 * F,
        'grid.color': GRID, 'grid.linewidth': 0.8 * F, 'grid.linestyle': '-',
        'legend.frameon': False, 'legend.fontsize': 9.5 * F,
        'figure.dpi': 110,
    })

def f(x):
    """A font size, line width or marker size written for the 9.5 base, scaled."""
    return x * F

def fs(x):
    """A scatter `s` (an area, points^2) written for the 9.5 base, scaled."""
    return x * F * F

def tidy(ax, yaxis_grid=True):
    """Hairline, recessive chrome. Solid gridlines only -- a dashed rule in
    these figures always means a reference level, never a grid."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    if yaxis_grid:
        ax.grid(axis='y', zorder=0)
    ax.set_axisbelow(True)

def dots(ax, x, vals, color, jitter, rng, marker='o', size=42, hollow=False,
         zorder=3, label=None):
    """Per-seed dots with a 2px surface ring so overlapping marks stay legible.
    Hollow = an encoder excluded from the median by a fixed rule: shown,
    never hidden, never counted."""
    j = rng.uniform(-jitter, jitter, len(vals))
    size = fs(size)
    ax.scatter(x + j, vals, s=size, marker=marker,
               facecolors=SURFACE if hollow else color,
               edgecolors=color if hollow else SURFACE,
               linewidths=f(1.4), zorder=zorder, label=label, alpha=0.98)
    if not hollow:
        size = fs(size)
    ax.scatter(x + j, vals, s=size, marker=marker, facecolors='none',
                   edgecolors=color, linewidths=f(0.9), zorder=zorder + 0.1)

def median_tick(ax, x, med, color, half=0.20, zorder=5):
    ax.plot([x - half, x + half], [med, med], color=color, lw=f(2.6),
            solid_capstyle='butt', zorder=zorder)


# ---------------------------------------------------------------------------
# Cross-figure geometry. Three quantities are
# held equal across every figure in the paper: the rendered point size, the
# whitespace between panels, and the offset of the panel label. The first is
# only exact when the saved file is the same width as the fig_w handed to
# apply(), so every figure saves with plain savefig at its own canvas; a
# bbox_inches='tight' save makes the file's width depend on its content and
# silently rescales the whole figure on the page.
LEGEND_Y        = 1.0      # figure-fraction y that every top legend anchors to
LABEL_OFFSET_PT = 38.0     # scaled points from the axes bottom to the panel label
LABEL_SIZE      = 12.5     # axes.labelsize is 10; cmr10 reads optically smaller
LABEL_FONT      = 'cmr10'  # Computer Modern Roman; tmlr.sty loads lmodern, which
                           # extends it, so this matches the document face
PANEL_GAP_IN    = 0.24     # rendered whitespace between neighbouring panels


def panel_label(ax, letter, text):
    """One format for every panel label: (a) text, centred beneath its panel.
    The offset is in scaled points, not a fraction of the axes height, so it is
    the same distance on the page whatever the figure's proportions."""
    ax.annotate(f'({letter}) {text}', xy=(0.5, 0.0), xycoords='axes fraction',
                xytext=(0, -f(LABEL_OFFSET_PT)), textcoords='offset points',
                ha='center', va='top', color=INK, fontsize=f(LABEL_SIZE),
                fontname=LABEL_FONT, annotation_clip=False)


def report(fig, name, fig_w):
    """Print the geometry that has to match across figures, and flag anything
    falling outside the canvas (which a tight save would have absorbed by
    enlarging the file). gap is measured between the panels' drawn extents,
    not their axes boxes: the right panel's y label lives inside that gap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axs = sorted([a for a in fig.get_axes() if a.get_subplotspec() is not None],
                 key=lambda a: a.get_position().x0)
    fw, fh = fig.get_size_inches()
    tbs = [a.get_tightbbox(r) for a in axs]
    gap = [(tbs[i + 1].x0 - tbs[i].x1) / fig.dpi * 6.5 / fw for i in range(len(axs) - 1)]
    tb = fig.get_tightbbox(r)
    sides = dict(left=max(0.0, -tb.x0), right=max(0.0, tb.x1 - fw),
                 bottom=max(0.0, -tb.y0), top=max(0.0, tb.y1 - fh))
    if max(sides.values()) > 0.005:
        print('  OVERFLOW in: ' + ' '.join(f'{k} {v:.3f}' for k, v in sides.items() if v > 0.005))
    print(f'REPORT {name} canvas {fw:.2f}x{fh:.2f} declared {fig_w:.2f} '
          f'gap_in {" ".join(f"{g:.3f}" for g in gap) or "-"} '
          f'font_pt {7.0 * fig_w / fw:.2f}')
