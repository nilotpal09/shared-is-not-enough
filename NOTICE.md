# Provenance and licensing

- **Training code** (`train.py`, `src/`) is a derivative of Meta AI's I-JEPA
  (https://github.com/facebookresearch/ijepa), released under Attribution-NonCommercial
  4.0 International (CC BY-NC 4.0); see `LICENSE`. The MAE* training path (`src/mae/`)
  is adapted from Meta AI's MAE. Our additions: the calorimeter data layer
  (`src/datasets/calo_flash.py`), the SimCLR* and BYOL* training loops and heads, the
  configurations, the probe code (`probes/`), the figure code (`figures/`) and the
  result tables (`results/`). They are released under the same CC BY-NC 4.0 terms.
- **PyCaloFlash** (the calorimeter shower simulator) is a separate dependency and is
  not included here. Install it from its repository at tag v1.0 and cite the
  archived release, Zenodo DOI 10.5281/zenodo.20588830. The jet generator and the
  shower-shape hook this code uses are part of v1.0.
