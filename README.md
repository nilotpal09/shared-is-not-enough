# Shared Is Not Enough: code

Code for **"Shared Is Not Enough: Predictor Width and Factor Readability Under
Masked Latent Prediction"** ([arXiv:2610.32562](https://arxiv.org/abs/2610.32562)). It trains the encoders, fits and scores
the readability probes, measures movement, and draws the figures. The trained
encoders and the fitted probes are on Zenodo:
[10.5281/zenodo.23559362](https://doi.org/10.5281/zenodo.23559362).

```
train.py            train one encoder (I-JEPA*, SimCLR*, BYOL*, MAE*)
configs/            one config per setting (30)
src/                training code: data layer, models, masking, the four training loops
probes/             probes, test-set scoring, movement, simulator facts
  runs.csv          every released checkpoint: setting, seed, tag, path in the Zenodo record
results/            the result tables behind the paper (see results/README.md)
figures/            one script per figure
```

## Install

Python 3.11 and a CUDA GPU. Versions used for the paper are in `requirements.txt`.

```
pip install -r requirements.txt
pip install "git+https://github.com/etiennedreyer/pycaloflash.git@v1.0"
```

PyCaloFlash, the calorimeter simulator, is a separate package: cite its archived
release, Zenodo DOI [10.5281/zenodo.20588830](https://doi.org/10.5281/zenodo.20588830).

## Checkpoints and probes

Download the Zenodo record and unzip it into `./checkpoints` (or anywhere, and
point `TESTBED_ROOT` or `--root` at it):

```
checkpoints/final/  projector_width/  trajectory/  reproducibility/  probes/
```

`probes/runs.csv` maps each setting and seed to its checkpoint file. Probe images
are generated from fixed seeds on first use and cached in `./cache`
(`TESTBED_CACHE` or `--cache` to move it).

## Reproduce the paper's numbers from the released files

```
python probes/score_test.py --tree npz_c40k/N1000/ps1 --probe-n 1000 --kind big   # test-set scores
python probes/movement.py --setting ijepa_shared_default --seeds 1 3 4 5 6 7       # movement
```

`score_test.py` rescores saved probes on the test set without refitting; the
folders under `probes/` in the record are listed in its docstring and match the
`group` column of `results/readability.csv`. On the GPU model the paper used
(NVIDIA TITAN RTX), both scripts reproduce the values in `results/` exactly.

## Fit new probes

```
python probes/fit_probes.py --setting ijepa_shared_wider --seed 4            # main probe, 40,000 steps
python probes/fit_probes.py --untrained 5 --probe-n 100                      # untrained reference
python probes/fit_probes.py --setting simclr_shared --seed 1 --kinds simple  # small probe
python probes/fit_z1.py --setting ijepa_shared_default --seed 3              # single-target probe
python probes/raw_input.py --variant raw_ln --probe-seed 1                   # raw input, no encoder
python probes/std_probe.py --setting simclr_shared --seed 1 --probe-seed 1   # standardised probe
```

## Train

```
python train.py --config configs/ijepa/ijepa_shared_default.yml --seed 3
```

Checkpoints, the loss log and a copy of the config go to
`runs/<config name>_seed<seed>/`. A file tagged `ep<N>` holds the state after
N - 1 completed epochs, so the final encoders (`ep500`) are the state after 499.
Training is not bitwise reproducible with the default GPU kernels; with
`meta.deterministic: true` (as in `configs/reproducibility/`), two same-seed runs
give identical checkpoints.

## Figures

| Figure | Script | Reads |
|---|---|---|
| 1 | `figures/view_pair_construction.tex` (TikZ) | |
| 3 | `figures/fig_core.py` | `results/readability.csv` |
| 4 | `figures/fig_support.py` | `results/readability.csv` |
| 5 | `figures/fig_movement_comparisons.py` | `results/readability.csv`, `results/movement.csv` |
| 6 | `figures/fig_trajectory_capacity.py` | `results/readability.csv` |
| 7 | `figures/fig_diet_curves.py` | `results/readability.csv` |

With the package versions in `requirements.txt`, the figures are identical to the
paper's.

## Licence and citation

CC BY-NC 4.0, as a derivative of Meta AI's I-JEPA; see `LICENSE` and `NOTICE.md`.

Please cite the paper:

```bibtex
@misc{kakati2026sharedenoughpredictorwidth,
      title={Shared Is Not Enough: Predictor Width and Factor Readability Under Masked Latent Prediction}, 
      author={Nilotpal Kakati},
      year={2026},
      eprint={2610.32562},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.32562}, 
}
```
