"""BYOL training loop — the second objective-class control (the ingredient split).

Reached through `train.py` for configs that set `meta.objective: byol`
(asserted below). It shares the data, model and optimizer code with the
I-JEPA loop (`src/train.py`).

What is held identical to the JEPA arms
---------------------------------------
* data: the same `make_calo` loader, the same `MaskCollator` doing the
  collation, the same config knobs. The mask tensors it produces are still
  produced (the collator is what batches the images) — the BYOL objective just
  does not consume them, which is the ingredient this arm removes. Producing
  them draws block sizes from the collator's own `torch.Generator` and block
  positions from the global RNG, exactly as on the JEPA path, so the global RNG
  stream advances as in a JEPA run.
* encoder: `src.helper.init_model` is called UNCHANGED, so the encoder's
  initialization consumes exactly the RNG a JEPA run at the same seed consumes
  (including the predictor construction that sits between encoder construction
  and encoder weight init). The JEPA predictor object is then dropped without
  being used; BYOL's own prediction head is a separate, smaller module built
  afterwards.
* the EMA: `copy.deepcopy(encoder)` for the target, `requires_grad = False` on
  its parameters, and the momentum schedule
  `m = ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)` — the same three
  lines `train.py` runs, on the same `optimization.ema: [0.996, 1.0]` config
  values. The house schedule and BYOL's own published tau_base = 0.996 rising to
  1.0 happen to coincide, so nothing had to be invented here.
* optimizer/schedules: AdamW with the same no-weight-decay split for bias/1-d
  params, the same `WarmupCosineSchedule` and `CosineWDSchedule` on the same
  config values. Both heads take the encoder's schedule.

What is different, and only this
--------------------------------
The objective. Instead of predicting EMA-target latents of MASKED BLOCKS, the
whole view is pooled, projected and predicted, and the online prediction of one
view regresses the EMA target's projection of the OTHER view (symmetrized), with
no negatives anywhere (`src/byol.py`).

Checkpoints
-----------
`ckpt['target_encoder']` is the EMA encoder and `ckpt['encoder']` the online one,
both in `train.py`'s exact layout (DDP `module.` prefix included). The v2 probe
calls `encoder_state_dict_from_ckpt(ckpt, prefer='target_encoder')`, so **it
reads the EMA encoder by default — the same side of the EMA it reads on every
JEPA arm**, which is what makes these runs comparable to them. The online encoder
stays available under `'encoder'` for a second readout. Both heads are stored
under their own keys (`projector`, `target_projector`, `predictor_head`) and are
never part of either encoder state dict.
"""

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.datasets.calo_flash import make_calo, n_channels_from_cfg, img_size_from_cfg, CaloNorm

from src.helper import init_model
from src.byol import (ProjectionHead, PredictionHead, pool_tokens, byol_loss,
                      momentum_update, momentum_schedule)

# --
log_timings = True
log_freq = 10
# --

_GLOBAL_SEED = 1
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_opt_byol(
    encoder,
    projector,
    predictor_head,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    """`src.helper.init_opt` with the JEPA predictor's two param groups replaced
    by the projection head's and the prediction head's. Same optimizer, same
    split rule, same schedules. The TARGET modules are deliberately absent: they
    are never optimized, only EMA'd."""
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in projector.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor_head.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in projector.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor_head.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    logger.info('Using AdamW (byol: encoder + projection head + prediction head)')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch))
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def main(args, resume_preempt=False, run_folder=None):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- OBJECTIVE GATE. This entrypoint refuses anything that is not explicitly
    #    marked byol, so a JEPA or contrastive config can never be silently
    #    retrained under a different loss.
    objective = args.get('meta', {}).get('objective')
    assert objective == 'byol', (
        f"this loop requires meta.objective: byol, got {objective!r}.")

    # -- META
    local_seed = args.get('meta', {}).get('seed', _GLOBAL_SEED)
    if local_seed != _GLOBAL_SEED:
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)

    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    checkpoint_freq = args['meta'].get('checkpoint_freq', 1)
    checkpoint_schedule = args['meta'].get('checkpoint_schedule', None)

    def _ckpt_due(ep1):  # ep1 is 1-indexed epoch just completed
        if checkpoint_schedule:
            prev = 0
            for until, every in checkpoint_schedule:
                if ep1 <= until:
                    return (ep1 - prev) % every == 0
                prev = until
            return ep1 % checkpoint_schedule[-1][1] == 0
        return ep1 % checkpoint_freq == 0

    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    # -- BYOL knobs
    b_cfg = args['meta'].get('byol') or {}
    proj_hidden_dim = b_cfg.get('proj_hidden_dim')
    proj_out_dim = b_cfg.get('proj_out_dim')
    pred_hidden_dim = b_cfg.get('pred_hidden_dim')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    if args.get('meta', {}).get('deterministic', False):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.cuda.manual_seed_all(local_seed)
        logger.info(f'[determinism] forced deterministic kernels (seed={local_seed})')

    # -- DATA (calo only: this control is a calo-family arm)
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_cfg = args['data'].get('calo_cfg')
    assert args['data'].get('sample_target', False), (
        'byol training needs the paired views: set data.sample_target: true')
    img_size = img_size_from_cfg(data_cfg)

    # -- MASK (produced by the collator exactly as on the JEPA path; NOT consumed
    #    here — dropping masking is the whole point of this arm)
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging'].get('folder', 'runs')
    tag = args['logging']['write_tag']
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- resolve run folder (the entry script passes one; otherwise a new one
    #    under logging.folder, named after the run)
    if run_folder is None:
        run_name = args['logging'].get('run_name') or f'{tag}_{time.strftime("%Y%m%d-%H%M%S")}'
        run_folder = os.path.join(folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    if rank == 0:
        with open(os.path.join(run_folder, 'config.yml'), 'w') as f:
            yaml.dump(args, f)

    log_file = os.path.join(run_folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(run_folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(run_folder, f'{tag}-latest.pth.tar')

    # -- csv logger: the JEPA schema. mask-A/mask-B are the sizes the collator
    #    produced; the objective ignores them.
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    in_chans = n_channels_from_cfg(data_cfg)
    # init_model is called UNCHANGED so the encoder init matches a JEPA run at the
    # same seed bit for bit; the JEPA predictor it also builds is dropped here and
    # never used. BYOL's own prediction head is built separately below.
    encoder, _unused_predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=img_size[0],
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        in_chans=in_chans)
    del _unused_predictor
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Head-width sweep (2026-09-24): with byol.rng_ref_proj_hidden_dim set, the
    # projector is built so the global RNG ends where a projector of the
    # reference width would leave it; the predictor below then starts as in the
    # reference run at the same seed. Unset (every earlier config): unchanged.
    rng_ref_proj_hidden_dim = b_cfg.get('rng_ref_proj_hidden_dim')
    if rng_ref_proj_hidden_dim is None or rng_ref_proj_hidden_dim == proj_hidden_dim:
        projector = ProjectionHead(encoder.embed_dim,
                                   hidden_dim=proj_hidden_dim,
                                   out_dim=proj_out_dim).to(device)
    else:
        from src.rng_match import rng_matched
        projector = rng_matched(
            lambda: ProjectionHead(encoder.embed_dim, hidden_dim=proj_hidden_dim,
                                   out_dim=proj_out_dim),
            lambda: ProjectionHead(encoder.embed_dim, hidden_dim=rng_ref_proj_hidden_dim,
                                   out_dim=proj_out_dim)).to(device)
        logger.info('[byol] projector RNG matched to hidden width %d',
                    rng_ref_proj_hidden_dim)
    predictor_head = PredictionHead(projector.out_dim,
                                    hidden_dim=pred_hidden_dim).to(device)
    logger.info('[byol] projector %d -> %d -> %d | predictor %d -> %d -> %d',
                projector.in_dim, projector.hidden_dim, projector.out_dim,
                predictor_head.dim, predictor_head.hidden_dim, predictor_head.dim)

    # -- the EMA twin: encoder AND projector, no predictor. Built by deepcopy so
    #    it starts identical to the online network (deepcopy consumes no RNG).
    target_encoder = copy.deepcopy(encoder)
    target_projector = copy.deepcopy(projector)

    mask_collator = MBMaskCollator(
        input_size=img_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    norm_cfg = args['data'].get('norm') or {}
    norm = CaloNorm(**norm_cfg) if norm_cfg else None

    dataset, unsupervised_loader, unsupervised_sampler = make_calo(
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        sample_target=args['data'].get('sample_target', False),
        persistent_workers=args['data'].get('persistent_workers', True),
        num_events=args['data'].get('num_events', 100_000),
        sampler_seed=local_seed,
        norm=norm,
        calo_cfg=data_cfg,
        generator_cfg=args['data'].get('generator_cfg'),
    )
    ipe = len(unsupervised_loader)

    optimizer, scaler, scheduler, wd_scheduler = init_opt_byol(
        encoder=encoder,
        projector=projector,
        predictor_head=predictor_head,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    projector = DistributedDataParallel(projector, static_graph=True)
    predictor_head = DistributedDataParallel(predictor_head, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    target_projector = DistributedDataParallel(target_projector)
    for p in target_encoder.parameters():
        p.requires_grad = False
    for p in target_projector.parameters():
        p.requires_grad = False

    # -- momentum schedule: verbatim from src/train.py (see src/byol.py)
    momentum_scheduler = momentum_schedule(ema, ipe, num_epochs, ipe_scale)

    start_epoch = 0

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),                    # ONLINE encoder
            'target_encoder': target_encoder.state_dict(),      # EMA encoder (probe default)
            'projector': projector.state_dict(),
            'target_projector': target_projector.state_dict(),
            'predictor_head': predictor_head.state_dict(),      # online side only
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'objective': 'byol',
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if _ckpt_due(epoch + 1):
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        unsupervised_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        cos_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        itr_start = time.time()
        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            data_time_ms = (time.time() - itr_start) * 1e3  # collate (sim) time

            view1 = udata[0].to(device, non_blocking=True)   # (B, C, H, W)
            assert udata[1].ndim == 4 and udata[1].shape == view1.shape, (
                'byol training needs two views per event; the loader emitted '
                f'udata[1] of shape {tuple(udata[1].shape)} against {tuple(view1.shape)}. '
                'Check data.sample_target.')
            view2 = udata[1].to(device, non_blocking=True)   # (B, C, H, W)
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            _diag = {}

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                B = view1.shape[0]
                views = torch.cat([view1, view2], dim=0)

                # -- TARGET branch: EMA encoder + EMA projector, NO predictor,
                #    entirely under no_grad. This is the stop-gradient.
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                        ht = target_encoder(views)                    # (2B, N, D)
                        t = target_projector(pool_tokens(ht))         # (2B, P)
                    t = t.float()
                    t1, t2 = t[:B], t[B:]

                # -- ONLINE branch: encoder + projector + PREDICTOR
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = encoder(views)                                # (2B, N, D)
                    z = projector(pool_tokens(h))                     # (2B, P)
                    p = predictor_head(z)                             # (2B, P)
                # The regression and its statistics run in fp32 even when the
                # encoder runs in bf16: cosines near 1 are exactly where bf16 is
                # worst, and the cost is negligible.
                with torch.amp.autocast('cuda', enabled=False):
                    loss, byol_stats = byol_loss(p[:B].float(), p[B:].float(), t1, t2)
                loss = AllReduce.apply(loss)

                _diag.update(byol_stats)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # -- momentum update of the target network: encoder AND projector,
                #    the same three lines src/train.py runs. `.detach()` on the
                #    online side, in-place on the target side, under no_grad.
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    _diag['ema_m'] = float(m)
                    for online, targ in ((encoder, target_encoder),
                                         (projector, target_projector)):
                        momentum_update(online, targ, m)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            cos_meter.update(_diag.get('byol_cos', 0.0))
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.4f (uncorrelated 2.000) cos: %.4f '
                                'embed-std: %.5f target-std: %.5f spread-frac: %.5f '
                                '[m: %.6f] [wd: %.2e] [lr: %.2e] '
                                '[mem cur/peak: %.2e / %.2e MB] '
                                '(data: %.1f ms | train: %.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   cos_meter.avg,
                                   _diag.get('byol_embed_std', float('nan')),
                                   _diag.get('byol_target_std', float('nan')),
                                   _diag.get('byol_spread_frac', float('nan')),
                                   _diag.get('ema_m', float('nan')),
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.memory_allocated() / 1024.**2,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   data_time_ms,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'
            itr_start = time.time()

        logger.info('avg. loss %.4f | avg. cos %.4f' % (loss_meter.avg, cos_meter.avg))
        # NOTE the off-by-one is the house convention, copied verbatim from
        # train.py: called with epoch+1, and the file tag is that +1 again, so
        # `jepa-ep{N}.pth.tar` holds the model after N-1 completed epochs and the
        # ep500 tag the probes read is written after epoch 499. Reproduced exactly
        # so these runs are checkpoint-comparable with every existing arm.
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
