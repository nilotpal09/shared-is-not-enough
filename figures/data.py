"""Read the result tables (results/*.csv) for the figure scripts."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / 'results'
# BYOL* encoders whose between-sample distance collapsed to zero; kept apart, never pooled
COLLAPSED = {('byol_shared', 4), ('byol_shared', 7), ('byol_constant', 6)}


@lru_cache(maxsize=None)
def _readability():
    return pd.read_csv(RESULTS / 'readability.csv')


@lru_cache(maxsize=None)
def movement():
    return pd.read_csv(RESULTS / 'movement.csv')


def readability(group='main', probe='big', probe_n=1000):
    d = _readability()
    return d[(d.group == group) & (d.probe == probe) & (d.probe_n == probe_n)]


def per_seed(df, setting, target='z1', keep_collapsed=True):
    """Test-set score per training seed: the mean of the two probe draws."""
    g = df[df.setting == setting].groupby('train_seed')[f'delta_{target}_test'].agg(['mean', 'size'])
    assert len(g) and set(g['size']) == {2}, (setting, sorted(set(g['size'])))
    v = g['mean']
    if keep_collapsed:
        return v
    return v[[(setting, int(s)) not in COLLAPSED for s in v.index]]


def collapsed(df, setting, target='z1'):
    v = per_seed(df, setting, target)
    return v[np.array([(setting, int(s)) in COLLAPSED for s in v.index], dtype=bool)]


# the paper's labels for the final encoders
LABEL = {
    'I-JEPA* shared': 'ijepa_shared_default', 'I-JEPA* constant': 'ijepa_constant_default',
    'I-JEPA* redrawn': 'ijepa_redrawn_default', 'I-JEPA* single-view': 'ijepa_singleview_default',
    'I-JEPA* deeper': 'ijepa_shared_deeper', 'I-JEPA* wider': 'ijepa_shared_wider',
    'SimCLR* shared': 'simclr_shared', 'SimCLR* constant': 'simclr_constant', 'SimCLR* redrawn': 'simclr_redrawn',
    'BYOL* shared': 'byol_shared', 'BYOL* constant': 'byol_constant', 'MAE*': 'mae_singleview',
    'untrained': 'untrained',
}


def seeds(name, target='z1', n=1000, keep_collapsed=False):
    """Main-probe test-set score per training seed for a paper label or setting
    name; collapsed BYOL* encoders left out unless keep_collapsed."""
    return per_seed(readability('main', 'big', n), LABEL.get(name, name), target, keep_collapsed)


# Figure 5 cohorts: (cohort name, setting); readability is the main probe's
# test-set score per seed, movement the per-token median at N = 1000
MOVEMENT_COHORTS = {'shared_default': 'ijepa_shared_default', 'shared_deeper': 'ijepa_shared_deeper',
                    'shared_wider': 'ijepa_shared_wider', 'shared_simclr': 'simclr_shared',
                    'constant_default': 'ijepa_constant_default', 'shared_byol': 'byol_shared'}


def movement_comparison_table():
    """One row per (cohort, training seed): readability, movement, collapsed."""
    mv = movement()
    mv = mv[mv['encoder_copy'] == 'default']
    rows = []
    for cohort, setting in MOVEMENT_COHORTS.items():
        r = per_seed(readability(), setting)
        m = mv[mv.setting == setting].set_index('train_seed').r_token_med
        for s in r.index:
            rows.append(dict(cohort=cohort, setting=setting, train_seed=int(s), readability=float(r[s]),
                             movement=float(m[s]), collapsed=int((setting, int(s)) in COLLAPSED)))
    return pd.DataFrame(rows)


def untrained_movement():
    """Median movement of the untrained inits (0.230)."""
    mv = movement()
    return float(mv[mv.setting == 'untrained'].r_token_med.median())
