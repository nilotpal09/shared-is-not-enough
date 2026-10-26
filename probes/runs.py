"""The released checkpoints (runs.csv) and how to load them.

runs.csv has one row per checkpoint in the Zenodo record: the paper's setting
name, training seed, checkpoint tag (jepa-ep<N> / mae-ep<N> hold the state after
N - 1 epochs), the file's path inside the record, the encoder copy the paper
probes, the training config, the original run folder name (which the result
CSVs and probe files refer to) and the setting's name in the result CSVs (arm).
"""
import csv
from functools import lru_cache
from pathlib import Path

import yaml

import common as C

RUNS_CSV = Path(__file__).resolve().parent / 'runs.csv'


@lru_cache(maxsize=1)
def runs():
    with open(RUNS_CSV) as fh:
        return tuple(csv.DictReader(fh))


def find(setting=None, seed=None, checkpoint_tag=None, run_folder=None, arm=None):
    """The one checkpoint row matching every given field."""
    want = dict(setting=setting, seed=None if seed is None else str(seed),
                checkpoint_tag=checkpoint_tag, run_folder=run_folder, arm=arm)
    hits = [r for r in runs() if all(v is None or r[k] == v for k, v in want.items())]
    if len(hits) != 1:
        raise KeyError(f'{len(hits)} checkpoints match {({k: v for k, v in want.items() if v})}')
    return hits[0]


def config(row):
    with open(C.REPO / 'configs' / row['config']) as fh:
        return yaml.safe_load(fh)


def load(row, root=None, prefer=None):
    """(encoder, key) for a runs.csv row; prefer defaults to the copy the paper probes."""
    prefer = prefer or ('encoder' if row['probed_key'] == 'encoder' else 'target_encoder')
    return C.load_encoder(C.data_root(root) / row['path'], config(row), prefer=prefer)
