"""Build one module while leaving the global CPU RNG where another would.

Added 2026-09-24 for the comparator-head width sweep (SimCLR* and BYOL*
projector hidden width 48 / 96 beside the reference 384). The projector is
built on the CPU from the global torch RNG, and it is built before the BYOL*
predictor and before the data pipeline. Changing its width changes how many
random numbers it consumes, so without this helper the predictor (and
everything drawn after it) would start differently from the reference run
at the same seed.

`rng_matched(make, make_ref)` returns `make()`, drawn from the same starting
RNG state as the reference module, and leaves the global CPU RNG exactly
where `make_ref()` would have left it. The reference module is built only to
advance the RNG and is discarded.
"""

import torch


def rng_matched(make, make_ref):
    before = torch.get_rng_state()
    make_ref()
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    module = make()
    torch.set_rng_state(after)
    return module
