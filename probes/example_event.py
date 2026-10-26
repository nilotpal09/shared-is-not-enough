"""The example event behind the graininess figure and the simulator facts: the
first of 80 seeds whose jet centroid lies closest to the detector centre,
simulated at several graininess levels with the same truth and shape draw."""
import torch

import common as C


def make_collator(n_spots, cfg=None):
    cfg = cfg or C.base_cfg()
    sub = {k: v for k, v in cfg['data']['calo_cfg'].items()
           if k not in ('N_spots_range', 'paired_fixed_long')}
    sub['N_spots_per_layer'] = int(n_spots)
    bc = C.CaloFlashBatchCollator(base_collator=lambda x: x, device=C.DEVICE,
                                  calo_cfg=sub, generator_cfg=cfg['data']['generator_cfg'],
                                  norm=C.CaloNorm())
    bc._build_backend()
    return bc


def find_centered_jet_seed(max_tries=80):
    """Search for a seed whose generated jet has centroid near the
    detector face (0, 0). Returns the seed."""
    bc = make_collator(1000)
    best_seed, best_dist = 0, 1e9
    for seed in range(max_tries):
        p_E, p_x, p_y = bc._generate_truth(batch_size=1, seed=seed)
        real = (~torch.isnan(p_E)) & (p_E > 0)
        E = torch.where(real, p_E, torch.zeros_like(p_E))
        total_E = E.sum().clamp(min=1.0)
        cx = (E * torch.where(real, p_x, torch.zeros_like(p_x))).sum() / total_E
        cy = (E * torch.where(real, p_y, torch.zeros_like(p_y))).sum() / total_E
        d = float((cx ** 2 + cy ** 2).sqrt().item())
        if d < best_dist:
            best_dist = d
            best_seed = seed
    return best_seed


def simulate_at_n_spots(seed, n_spots_list):
    """[(N, depth-summed raw image (64, 64))]: same truth and same shape draw at every N."""
    panels = []
    for N in n_spots_list:
        bc = make_collator(N)
        p_E, p_x, p_y = bc._generate_truth(batch_size=1, seed=seed)
        img_xyz = bc._simulate_once(p_E, p_x, p_y, bc._sim_seed(seed, 0))
        panels.append((N, img_xyz[0].sum(dim=0).detach().cpu().numpy()))
    return panels
