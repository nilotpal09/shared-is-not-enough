"""Contrastive (InfoNCE) training loop — the objective-class control arm.

Reached through `train.py` for configs that set `meta.objective: contrastive`
(the paper's SimCLR*; asserted below). It shares the data, model and optimizer
code with the I-JEPA loop (`src/train.py`).

What is held identical to the JEPA arms
---------------------------------------
* data: the same `make_calo` loader, the same `MaskCollator` doing the collation,
  the same config knobs. The mask tensors it produces are still produced (the
  collator is what batches the images) — the contrastive objective just does not
  consume them. Producing them draws block sizes from the collator's own
  `torch.Generator` and block positions from the global RNG, exactly as on the
  JEPA path, so the global RNG stream advances as in a JEPA run.
* encoder: `src.helper.init_model` is called UNCHANGED, so the encoder's
  initialization consumes exactly the RNG a JEPA run at the same seed consumes
  (including the predictor construction that sits between encoder construction
  and encoder weight init). The predictor object is then dropped without being
  used, and no target encoder is ever built — so no EMA, no predictor, no
  momentum schedule runs here.
* optimizer/schedules: AdamW with the same no-weight-decay split for
  bias/1-d params, the same `WarmupCosineSchedule` and `CosineWDSchedule` on the
  same config values. The projection head takes the encoder's schedule.

What is different, and only this
--------------------------------
The loss. Instead of predicting EMA-target latents of masked blocks, the two
views are pooled, projected, and scored against the batch with NT-Xent
(`src/contrastive.py`).

Checkpoints keep the JEPA layout for the keys the probes read: `ckpt['encoder']`
is the plain encoder state dict (DDP `module.` prefix included, exactly as
`train.py` writes it). There is deliberately NO `target_encoder` key, so
`encoder_state_dict_from_ckpt(ckpt, prefer='target_encoder')` falls through to
`ckpt['encoder']` and the v2 probe protocol runs unchanged. The projection head
is saved under its own `projector` key and is not part of the encoder.
"""

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

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
from src.contrastive import ProjectionHead, pool_tokens, info_nce

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


def init_opt_contrastive(
    encoder,
    projector,
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
    """`src.helper.init_opt` with the predictor's two param groups replaced by
    the projection head's. Same optimizer, same split rule, same schedules."""
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in projector.named_parameters()
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
        }
    ]
    logger.info('Using AdamW (contrastive: encoder + projection head)')
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
    #    marked contrastive, so a JEPA config can never be silently retrained
    #    under a different loss (and vice versa: the JEPA entrypoint is the one
    #    these configs are never pointed at).
    objective = args.get('meta', {}).get('objective')
    assert objective == 'contrastive', (
        f"this loop requires meta.objective: contrastive, got {objective!r}.")

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

    # -- CONTRASTIVE knobs
    c_cfg = args['meta'].get('contrastive') or {}
    temperature = float(c_cfg.get('temperature', 0.2))
    assert temperature > 0, f'meta.contrastive.temperature must be > 0, got {temperature}'
    proj_hidden_dim = c_cfg.get('proj_hidden_dim')
    proj_out_dim = c_cfg.get('proj_out_dim')

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

    # -- DATA (calo only: the contrastive control is a calo-family arm)
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_cfg = args['data'].get('calo_cfg')
    assert args['data'].get('sample_target', False), (
        'contrastive training needs the paired views: set data.sample_target: true')
    img_size = img_size_from_cfg(data_cfg)

    # -- MASK (produced by the collator exactly as on the JEPA path; unused here)
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -- OPTIMIZATION
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
    # init_model is called UNCHANGED so the encoder init matches a JEPA run at
    # the same seed bit for bit; the predictor it also builds is dropped here and
    # never used, and no target encoder / EMA is created at all.
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

    # Head-width sweep (2026-09-24): with contrastive.rng_ref_proj_hidden_dim
    # set, the projector is built so the global RNG ends where a projector of
    # the reference width would leave it, so everything drawn after it starts
    # as in the reference run at the same seed. Unset (every earlier config):
    # unchanged.
    rng_ref_proj_hidden_dim = c_cfg.get('rng_ref_proj_hidden_dim')
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
        logger.info('[contrastive] projector RNG matched to hidden width %d',
                    rng_ref_proj_hidden_dim)
    logger.info('[contrastive] projection head %d -> %d -> %d, temperature %.4g',
                projector.in_dim, projector.hidden_dim, projector.out_dim, temperature)

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

    optimizer, scaler, scheduler, wd_scheduler = init_opt_contrastive(
        encoder=encoder,
        projector=projector,
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

    start_epoch = 0

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'projector': projector.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'objective': 'contrastive',
            'temperature': temperature,
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
        acc_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        itr_start = time.time()
        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            data_time_ms = (time.time() - itr_start) * 1e3  # collate (sim) time

            view1 = udata[0].to(device, non_blocking=True)   # (B, C, H, W)
            assert udata[1].ndim == 4 and udata[1].shape == view1.shape, (
                'contrastive training needs two views per event; the loader emitted '
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
                # One forward on the concatenated views: identical to two separate
                # forwards (no batch-coupled op in the encoder), and it keeps DDP
                # to one forward per backward.
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = encoder(torch.cat([view1, view2], dim=0))    # (2B, N, D)
                    p = projector(pool_tokens(h))                    # (2B, P)
                # The similarity matrix and its cross-entropy run in fp32 even when
                # the encoder runs in bf16: a 2Bx2B softmax in bf16 is needlessly
                # lossy and the cost is negligible.
                with torch.amp.autocast('cuda', enabled=False):
                    loss, nce_stats = info_nce(p[:B].float(), p[B:].float(),
                                               temperature=temperature)
                loss = AllReduce.apply(loss)

                _diag.update(nce_stats)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            acc_meter.update(_diag.get('nce_acc', 0.0))
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f (chance %.3f) acc: %.3f '
                                'pos/neg: %.3f %.3f embed-std: %.4f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem cur/peak: %.2e / %.2e MB] '
                                '(data: %.1f ms | train: %.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   _diag.get('nce_chance_loss', float('nan')),
                                   acc_meter.avg,
                                   _diag.get('nce_pos_sim', float('nan')),
                                   _diag.get('nce_neg_sim', float('nan')),
                                   _diag.get('nce_embed_std', float('nan')),
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

        logger.info('avg. loss %.3f | avg. nce acc %.3f' % (loss_meter.avg, acc_meter.avg))
        # NOTE the off-by-one is the house convention, copied verbatim from
        # train.py: called with epoch+1, and the file tag is that +1 again, so
        # `jepa-ep{N}.pth.tar` holds the model after N-1 completed epochs and the
        # ep500 tag the probes read is written after epoch 499. Reproduced exactly
        # so these runs are checkpoint-comparable with every existing arm.
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
