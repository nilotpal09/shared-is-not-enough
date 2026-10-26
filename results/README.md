# Result tables

The probe scores, movement values and simulator facts behind the paper, one
table each; the figure scripts (`figures/`) draw every data figure from them.
Setting names are those of the configs (`configs/<method>/<setting>.yml`)
and of the Zenodo record; `untrained` is a randomly initialised encoder whose
`train_seed` is its initialisation seed.

## readability.csv: one row per probe fit (1,560 rows)

The test-set score of each probe behind the paper, at the step its selection set
chose. Rescoring the released probe files with `probes/score_test.py` gives the
same values.

| Column | Meaning |
|---|---|
| `setting`, `train_seed` | the encoder |
| `checkpoint_tag`, `completed_epochs` | `jepa-ep<N>` / `mae-ep<N>` holds the state after N - 1 epochs |
| `encoder_key`, `encoder_copy` | the weights read: `target_encoder` (EMA copy), `encoder` (online copy), `model(online)` (MAE*); `default` is the copy the paper probes, `online` the other one |
| `group` | `main` (main probe, final checkpoints), `online copy`, `small probe`, `single target`, `training over time`, `projector width` |
| `probe` | `big` (main probe, 85,189 parameters), `simple` (small probe, 7,811), `z1big` (single target, z1 only, 85,059) |
| `probe_n` | probe graininess: spots per layer of the probe images |
| `probe_seed` | probe draw (1 or 2); the paper averages the two per training seed |
| `steps`, `best_step` | probe training steps (40,000); step chosen on the selection set |
| `delta_<t>_test` | **the paper's score**: `kn_<t>_test - read_<t>_test`, for t = z1, z2, E |
| `read_<t>_test`, `kn_<t>_test` | median absolute error of the probe and of the constant prediction, test set |
| `delta_<t>_sel`, `read_<t>_sel`, `kn_<t>_sel` | the same on the selection set |
| `n_sel_particles`, `n_test_particles` | particles in the selection and test sets |
| `arm`, `run_folder` | the setting's internal name and original run folder (as in the probe files) |

The single-target rows have z1 columns only.

## movement.csv: one row per encoder (159 rows)

Per-token movement at N = 1000 (64 truths x 8 replicas that redraw only shower
shape): `r_token_med` (median over the 256 token positions; the paper's
movement), `r_token_q1`, `r_token_q3`, and `r_pooled` (mean-pooled tokens).
`probes/movement.py` reproduces every value exactly on the same GPU model
(`gpu`); other GPUs agree to within their arithmetic.

## raw_input.csv

The main probe on the raw input patches with no encoder (`raw_ln`: the probe as
it is; `raw_noln`: its per-token normalisation replaced by a fixed one), with
selection and test scores. From `probes/raw_input.py`.

## standardised_probe.csv: one row per probe fit (84 rows)

The main probe with its per-token LayerNorm replaced by a fixed per-dimension
standardisation computed on the fitting set (Table 4's standardised row and
Appendix B.4), on the final shared encoders (BYOL* without its two collapsed
ones), MAE* and eight untrained initialisations, probe draws 1 and 2. From
`probes/std_probe.py`. Columns as in `readability.csv`; `gpu` is the GPU model
of each fit.

## simulator_facts.csv

`fact, n_spots, statistic, value` from `probes/simulator_facts.py`: shower-maximum
depth, cell-energy range of the model input, and the non-empty cell counts of the
graininess figure.
