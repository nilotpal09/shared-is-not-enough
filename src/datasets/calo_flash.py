"""
PyCaloFlash-backed calorimeter loader for I-JEPA / SimCLR / BYOL / MAE training.

Batches are generated on the fly with PyCaloFlash (`torch_calo_flash`, installed
separately). The loader yields:
    ((imgs, labels), masks_enc, masks_pred)       -- single view
    ((ctx_imgs, tgt_imgs), masks_enc, masks_pred) -- two views (sample_target=True)

Images have shape (B, N_cells_z, N_cells_x, N_cells_y): all z-layers are used
as input channels.  The number of model input channels equals calo_cfg['N_cells_z'].

With two views the truth particles are simulated twice (different RNG seeds),
giving two detector responses of the same event. How shower shape (alpha, T)
relates the two views is set in calo_cfg:
    redrawn  (default)              each view draws its own shape
    shared   paired_fixed_long: true  one shape draw for both views
    constant frozen_alpha_t: [a, T]   one fixed shape for every particle
Graininess is N_spots_per_layer, or log-uniform in N_spots_range (mixed).
The training loop detects two views via the shape of udata[1].

For inference, call CaloFlashBatchCollator.generate_replicas() directly.
"""

from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from torch_calo_flash.calorimeter import CaloBlock
from torch_calo_flash.calo_flash import get_longitudinal_parameters
from torch_calo_flash.generator import EventGenerator, JetEventGenerator

_logger = getLogger()

_DEFAULT_N_CELLS_Z = 8
_DEFAULT_N_CELLS_XY = 64

class CaloNorm:
    """Log-compress calorimeter cell energies for ViT training.

    forward : raw MeV  -> normalized (model input)
    inverse : normalized -> raw MeV  (for plotting / physics analysis)

    Modes
    -----
    'none'
        Identity — no normalization.
    'log1p_scale'  (default)
        y = log1p(x) / log1p(scale)
        Maps [0, scale] roughly to [0, 1] with log compression.
        scale (MeV) should be set near the observed max single-cell energy.
        Default 5500 MeV covers the ~5426 MeV max seen in profiling runs.
        Empty cells (below threshold) map to 0; hit cells land in ~[0.37, 1.0].
    'log1p_standardize'
        y = (log1p(x) - mean) / std
        Requires precomputed mean / std over log1p(cell_e).
        Set mean and std after running estimate_log1p_stats().
    """

    MODES = ('none', 'log1p_scale', 'log1p_standardize')

    def __init__(
        self,
        mode: str = 'log1p_scale',
        scale: float = 5500.0,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f'CaloNorm mode must be one of {self.MODES}, got {mode!r}')
        self.mode = mode
        self.scale = float(scale)
        self._log1p_scale = float(torch.log1p(torch.tensor(scale)))
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return x
        x = torch.log1p(x)
        if self.mode == 'log1p_scale':
            return x / self._log1p_scale
        return (x - self.mean) / self.std  # log1p_standardize

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Return raw MeV from a normalized tensor."""
        if self.mode == 'none':
            return y
        if self.mode == 'log1p_scale':
            return torch.expm1(y * self._log1p_scale).clamp_(min=0.0)
        return torch.expm1(y * self.std + self.mean).clamp_(min=0.0)  # log1p_standardize


def _as_device(device):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    return device


def _default_calo_cfg(calo_cfg=None):
    default_cfg = {
        'Z': 82,
        'width': 10.0,
        'height': 10.0,
        'depth': 16.0,
        'N_cells_x': _DEFAULT_N_CELLS_XY,
        'N_cells_y': _DEFAULT_N_CELLS_XY,
        'N_cells_z': _DEFAULT_N_CELLS_Z,
        'N_spots_per_layer': 1000,
        'cell_e_threshold': 25.0,
    }
    if calo_cfg:
        default_cfg.update(calo_cfg)
    if default_cfg.get('height') is None:
        default_cfg['height'] = default_cfg['width']
    return default_cfg


def _default_generator_cfg(calo_cfg, generator_cfg=None):
    default_cfg = {
        'x_range': (-calo_cfg['width'] * 0.8 / 2, calo_cfg['width'] * 0.8 / 2),
        'y_range': (-calo_cfg['height'] * 0.8 / 2, calo_cfg['height'] * 0.8 / 2),
        'E_range': (1000.0, 100_000.0),
        'N_range': (4, 32),
        'pad_value': float('nan'),
    }
    if generator_cfg:
        default_cfg.update(generator_cfg)
    return default_cfg


def n_channels_from_cfg(calo_cfg: Optional[Dict[str, Any]] = None) -> int:
    """Return the number of image channels (= N_cells_z) for a given calo_cfg."""
    if calo_cfg and 'N_cells_z' in calo_cfg:
        return int(calo_cfg['N_cells_z'])
    return _DEFAULT_N_CELLS_Z


def img_size_from_cfg(calo_cfg: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """Return (N_cells_x, N_cells_y) — the spatial image dimensions.

    Raises ValueError if N_cells_x != N_cells_y, since the ViT patch embedding
    only supports square inputs.
    """
    nx = int((calo_cfg or {}).get('N_cells_x', _DEFAULT_N_CELLS_XY))
    ny = int((calo_cfg or {}).get('N_cells_y', _DEFAULT_N_CELLS_XY))
    if nx != ny:
        raise ValueError(
            f'N_cells_x ({nx}) != N_cells_y ({ny}): the ViT patch embedding '
            'requires square images. Use equal cell counts or add a non-square ViT.')
    return nx, ny


class VirtualCaloDataset(Dataset):
    """Index source so batch generation can happen in the collate function."""

    def __init__(self, n_events: int):
        super().__init__()
        self.n_events = int(n_events)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        return int(idx)


class CaloFlashBatchCollator:
    """Generate a full pycaloflash batch and delegate masking to the base collator."""

    def __init__(
        self,
        base_collator,
        sample_target: bool = False,
        device=None,
        calo_cfg: Optional[Dict[str, Any]] = None,
        generator_cfg: Optional[Dict[str, Any]] = None,
        norm: Optional['CaloNorm'] = None,
    ):
        self.base_collator = base_collator
        self.sample_target = sample_target
        self.device = _as_device(device)
        self.calo_cfg = _default_calo_cfg(calo_cfg=calo_cfg)
        self.generator_cfg = _default_generator_cfg(self.calo_cfg, generator_cfg=generator_cfg)
        self.n_channels = self.calo_cfg['N_cells_z']
        self.norm = norm if norm is not None else CaloNorm()
        self._generator = None
        self._calorimeter = None

    def _fork_devices(self):
        if self.device.type != 'cuda':
            return []
        if self.device.index is not None:
            return [self.device.index]
        return [torch.cuda.current_device()]

    def _build_backend(self):
        if self._generator is None:
            kind = self.generator_cfg.get('kind', 'gun')
            if kind == 'gun':
                self._generator = EventGenerator(self.generator_cfg, device=self.device)
            elif kind == 'jet':
                self._generator = JetEventGenerator(self.generator_cfg, device=self.device)
            else:
                raise ValueError(
                    f'unknown generator kind: {kind!r}. expected "gun" or "jet".')
        if self._calorimeter is None:
            self._calorimeter = CaloBlock(self.calo_cfg, device=self.device)

    @staticmethod
    def _batch_seed(batch):
        values = torch.as_tensor(batch, dtype=torch.long)
        weights = torch.arange(1, values.numel() + 1, dtype=torch.long)
        seed = int((values * weights).sum().item())
        return seed & 0x7FFFFFFF

    def _generate_truth(self, batch_size: int, seed: int):
        """Generate truth particles (p_E, p_x, p_y) reproducibly from seed."""
        self._build_backend()
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            p_E, p_x, p_y = self._generator.generate(batch_size)
        return p_E, p_x, p_y

    @staticmethod
    def _sample_n_spots_log_uniform(rng_pair, seed):
        """Sample N_spots log-uniformly in [low, high] using a deterministic
        per-batch RNG so the same batch always gets the same N_spots."""
        import numpy as np
        lo, hi = float(rng_pair[0]), float(rng_pair[1])
        if lo == hi:
            return int(round(lo))
        rs = np.random.RandomState(seed & 0xFFFFFFFF)
        n = int(round(np.exp(rs.uniform(np.log(lo), np.log(hi)))))
        return max(n, 1)

    def _resolve_n_spots(self, seed):
        """N_spots for this batch: log-uniform in N_spots_range when set (mixed
        graininess), else N_spots_per_layer. None falls back to the simulator's
        construction-time value."""
        rng_val = self.calo_cfg.get('N_spots_range')
        if rng_val:
            return self._sample_n_spots_log_uniform(rng_val, seed)
        scalar_val = self.calo_cfg.get('N_spots_per_layer')
        if scalar_val is not None:
            return int(scalar_val)
        return None

    def _compute_long_params(self, p_E, long_seed: int):
        """Pre-compute alpha/T longitudinal shower parameters under a dedicated seed.

        Returns the long_params dict matching what shoot() would compute internally,
        so a caller can hold shower SHAPE fixed across simulations while the spot
        positions are redrawn.

        Must replicate `simulate`'s flattening + masking exactly so the returned
        per-particle alpha/T values align with the real particles `shoot` will see.
        """
        flat_Es = p_E.reshape(-1)
        real_idx = torch.where(flat_Es == flat_Es)[0]   # NaN-aware: keeps non-pad
        real_Es = flat_Es[real_idx].to(device=self.device, dtype=torch.float32)
        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(long_seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(long_seed)
            return get_longitudinal_parameters(real_Es, self.calo_cfg['Z'])

    def _frozen_long_params(self, p_E, alpha0, T0):
        """Constant shape: the same (alpha, T) for EVERY particle in EVERY event,
        regardless of energy, so the encoder never sees shape vary."""
        from torch_calo_flash.calo_flash import get_beta
        lp = self._compute_long_params(p_E, 0)   # correct per-real-particle structure
        a = torch.full_like(lp['alpha'], float(alpha0))
        t = torch.full_like(lp['T'], float(T0))
        lp['alpha'], lp['T'], lp['beta'] = a, t, get_beta(a, t)
        lp['mean_ln_alpha'] = torch.log(a)
        return lp

    def _simulate_once(self, p_E, p_x, p_y, seed: int,
                       n_spots_override: Optional[int] = None,
                       long_seed: Optional[int] = None,
                       long_params_override: Optional[dict] = None) -> torch.Tensor:
        """Run one stochastic detector simulation.

        Returns (B, N_cells_z, N_cells_x, N_cells_y) — all z-layers as channels.
        If n_spots_override is given, uses that value instead of the simulator's
        construction-time N_spots_per_layer.
        If long_seed is given, the alpha/T longitudinal parameters are drawn under
        long_seed (so they're deterministic across calls that pass the same
        long_seed), while spot positions are still drawn from the main `seed`.
        If long_params_override is given (a dict from _compute_long_params,
        optionally edited per-particle), it is passed straight to the simulator
        and takes precedence over long_seed.
        """
        long_params = long_params_override
        if long_params is None and long_seed is not None:
            long_params = self._compute_long_params(p_E, long_seed)

        with torch.random.fork_rng(devices=self._fork_devices()):
            torch.manual_seed(seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            sim_kwargs = dict(return_grid=True, return_hits=False, return_truth=False)
            if n_spots_override is not None:
                sim_kwargs['N_spots_per_layer'] = n_spots_override
            if long_params is not None:
                sim_kwargs['long_params'] = long_params
            calo_dict = self._calorimeter.simulate(
                p_E, p_x, p_y,
                **sim_kwargs,
            )

        grid = calo_dict['cell_e']  # (B, N_cells_x, N_cells_y, N_cells_z)

        # Defensive safety net: replace any NaN/Inf cells with 0. With pad_value=NaN
        # this should be a no-op; it keeps a rare numerical edge case in the
        # simulator from crashing a 500-epoch run.
        bad = ~torch.isfinite(grid)
        if bad.any():
            grid = torch.where(bad, torch.zeros_like(grid), grid)

        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        # move z-axis (last) to channel position -> (B, N_cells_z, N_cells_x, N_cells_y)
        return grid.permute(0, 3, 1, 2).contiguous()

    def _sim_seed(self, truth_seed: int, replica_idx: int) -> int:
        """Derive a simulation seed well-separated from the truth seed."""
        return (truth_seed * 31337 + replica_idx + 1) & 0x7FFFFFFF

    def __call__(self, batch):
        batch_size = len(batch)
        seed = self._batch_seed(batch)
        p_E, p_x, p_y = self._generate_truth(batch_size, seed)
        n = self._resolve_n_spots(seed)

        if self.sample_target:
            frozen = self.calo_cfg.get('frozen_alpha_t')   # [alpha0, T0]: constant shape
            if frozen is not None:
                lp_frozen = self._frozen_long_params(p_E, frozen[0], frozen[1])
                kw = dict(long_params_override=lp_frozen)
            else:
                # shared: one alpha/T draw for both views (the long seed still varies
                # per batch, so different events get different shapes); redrawn: each
                # view draws its own.
                paired_long = (seed * 7919 + 31337) & 0x7FFFFFFF \
                    if self.calo_cfg.get('paired_fixed_long') else None
                kw = dict(long_seed=paired_long)
            imgs_ctx = self.norm.forward(self._simulate_once(
                p_E, p_x, p_y, self._sim_seed(seed, 0), n_spots_override=n, **kw))
            imgs_tgt = self.norm.forward(self._simulate_once(
                p_E, p_x, p_y, self._sim_seed(seed, 1), n_spots_override=n, **kw))
            sample_batch = [(imgs_ctx[i], imgs_tgt[i]) for i in range(batch_size)]
        else:
            imgs = self.norm.forward(self._simulate_once(
                p_E, p_x, p_y, self._sim_seed(seed, 0), n_spots_override=n))
            sample_batch = [(imgs[i], 0) for i in range(batch_size)]

        return self.base_collator(sample_batch)

    def generate_replicas(
        self,
        batch_size: int,
        n_replicas: int = 25,
        seed: Optional[int] = None,
        normalize: bool = False,
        fixed_long_seed: Optional[int] = None,
        fixed_spot_seed: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Inference helper: n_replicas independent detector responses for the same truth.

        Args:
            normalize: apply self.norm.forward() to replicas (False by default so
                       callers get raw MeV for physics analysis / plotting).
                       Use self.norm.inverse() to go back to MeV from a normalized tensor.
            fixed_long_seed: if set, all replicas use this single seed for the α/T
                       longitudinal draw, so the shower SHAPE is fixed per particle
                       across replicas and only the spot-level Monte Carlo varies.
                       Used for the MC-only ablation (isolates Stage-2). Default None
                       reproduces full stochasticity (both shape and spots vary).
            fixed_spot_seed: the MC-only ablation's mirror image (isolates Stage-1).
                       If set, all replicas share this single spot-sampling seed
                       (common random numbers) while the α/T draw VARIES per replica.
                       Within-truth variation is then driven only by α/T. Mutually
                       exclusive with fixed_long_seed. Used for the α/T-only probe.

        Returns:
            truth    -- (p_E, p_x, p_y) each shape (B, N_particles)
            replicas -- shape (B, n_replicas, N_cells_z, N_cells_x, N_cells_y)
        """
        if seed is None:
            seed = int(torch.randint(0, 0x7FFFFFFF, (1,)).item())
        self._build_backend()

        p_E, p_x, p_y = self._generate_truth(batch_size, seed)
        if fixed_spot_seed is not None:
            assert fixed_long_seed is None, \
                'fixed_spot_seed (α/T-only) and fixed_long_seed (MC-only) are mutually exclusive'
            # α/T-only: spot RNG shared across replicas (CRN); α/T draw varies per replica.
            replicas = [
                self._simulate_once(
                    p_E, p_x, p_y, fixed_spot_seed,
                    long_seed=self._sim_seed(seed, k),
                )
                for k in range(n_replicas)
            ]
        else:
            replicas = [
                self._simulate_once(
                    p_E, p_x, p_y, self._sim_seed(seed, k),
                    long_seed=fixed_long_seed,
                )
                for k in range(n_replicas)
            ]
        stacked = torch.stack(replicas, dim=1)
        if normalize:
            stacked = self.norm.forward(stacked)
        return (p_E, p_x, p_y), stacked


def make_calo(
    batch_size=64,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    training=True,
    drop_last=True,
    sample_target: bool = False,
    persistent_workers: bool = True,
    num_events: int = 100_000,
    device=None,
    calo_cfg: Optional[Dict[str, Any]] = None,
    generator_cfg: Optional[Dict[str, Any]] = None,
    norm: Optional[CaloNorm] = None,
    sampler_seed: int = 0,
):
    """
    Build a synthetic calorimeter dataset and DataLoader for training.

    Image channels correspond to all N_cells_z z-layers of the calorimeter.
    Use n_channels_from_cfg(calo_cfg) to get the channel count before dataset creation.

    When sample_target=True the collator produces paired (context, target) images
    from independent detector simulations of the same truth particles.

    For inference use CaloFlashBatchCollator.generate_replicas() directly.
    """
    if collator is None:
        raise ValueError('make_calo requires a mask collator')

    device = _as_device(device)
    if device.type == 'cuda':
        if num_workers > 0:
            _logger.warning(
                'pycaloflash GPU backend: forcing num_workers=0 '
                'to avoid multiple CUDA contexts.')
            num_workers = 0
            persistent_workers = False
        if pin_mem:
            _logger.warning(
                'pycaloflash GPU backend: forcing pin_memory=False — '
                'batches are already on CUDA, pinning would error.')
            pin_mem = False

    dataset = VirtualCaloDataset(n_events=num_events)
    batch_collator = CaloFlashBatchCollator(
        base_collator=collator,
        sample_target=sample_target,
        device=device,
        calo_cfg=calo_cfg,
        generator_cfg=generator_cfg,
        norm=norm,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        seed=sampler_seed,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=batch_collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    img_size = img_size_from_cfg(calo_cfg)
    _logger.info(
        'CaloFlash dataset created (N=%d, img_size=%dx%d, N_cells_z=%d, batch_size=%d, device=%s, sample_target=%s)',
        len(dataset),
        img_size[0],
        img_size[1],
        batch_collator.n_channels,
        batch_size,
        device,
        sample_target,
    )
    _logger.info(
        'CaloFlash data loader created (num_workers=%d, world_size=%d)',
        num_workers,
        world_size,
    )

    return dataset, data_loader, dist_sampler
