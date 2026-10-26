# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import sys
import time

import numpy as np
import yaml

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from src.masks.random import MaskCollator as RandomMaskCollator
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.datasets.calo_flash import make_calo, n_channels_from_cfg, img_size_from_cfg, CaloNorm

from src.mae.helper import init_mae_model, init_mae_opt, load_mae_checkpoint

# --
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False, run_folder=None):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    # per-run seed override (default falls back to the module-level _GLOBAL_SEED)
    local_seed = args.get('meta', {}).get('seed', _GLOBAL_SEED)
    if local_seed != _GLOBAL_SEED:
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)

    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    decoder_depth = args['meta']['decoder_depth']
    decoder_embed_dim = args['meta']['decoder_embed_dim']
    decoder_num_heads = args['meta']['decoder_num_heads']
    normalize_pixels = args['meta'].get('normalize_pixels', True)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    data_cfg = args['data'].get('calo_cfg')
    img_size = img_size_from_cfg(data_cfg)  # (H, W); raises if non-square

    # -- MASK
    patch_size = args['mask']['patch_size']
    mask_ratio = args['mask']['mask_ratio']

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

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
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

    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'mask-ratio'),
        ('%d', 'time (ms)'),
    )


    # -- init model
    in_chans = n_channels_from_cfg(data_cfg)
    model = init_mae_model(
        device=device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=img_size[0],
        decoder_depth=decoder_depth,
        decoder_embed_dim=decoder_embed_dim,
        decoder_num_heads=decoder_num_heads,
        normalize_pixels=normalize_pixels,
        in_chans=in_chans,
    )

    mask_collator = RandomMaskCollator(
        ratio=(mask_ratio, mask_ratio),
        input_size=img_size,
        patch_size=patch_size,
    )

    norm_cfg = args['data'].get('norm') or {}
    norm = CaloNorm(**norm_cfg) if norm_cfg else None

    _, unsupervised_loader, unsupervised_sampler = make_calo(
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        persistent_workers=args['data'].get('persistent_workers', True),
        num_events=args['data'].get('num_events', 100_000),
        norm=norm,
        calo_cfg=data_cfg,
        generator_cfg=args['data'].get('generator_cfg'),
    )
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_mae_opt(
        model=model,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    model = DistributedDataParallel(model, static_graph=True)

    start_epoch = 0
    if load_model:
        model, optimizer, scaler, start_epoch = load_mae_checkpoint(
            device=device,
            r_path=load_path,
            model=model,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            mask_collator.step()

    # Optional dense checkpointing, same convention as src/train.py:
    # meta.checkpoint_schedule = [[until_epoch, every], ...]. Absent -> the
    # pre-existing every-checkpoint_freq behavior, unchanged (2026-08-13,
    # Study D salvage: MAE was never checkpointed before ep50).
    checkpoint_schedule = args['meta'].get('checkpoint_schedule', None)

    def _ckpt_due(ep1):
        if checkpoint_schedule:
            prev = 0
            for until, every in checkpoint_schedule:
                if ep1 <= until:
                    return (ep1 - prev) % every == 0
                prev = until
            return ep1 % checkpoint_schedule[-1][1] == 0
        return ep1 % checkpoint_freq == 0

    def save_checkpoint(epoch):
        save_dict = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
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
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)  # (B, C, H, W)
                me = [u.to(device, non_blocking=True) for u in masks_enc]
                mp_ = [u.to(device, non_blocking=True) for u in masks_pred]
                return imgs, me, mp_

            imgs, me, mp_ = load_imgs()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    loss, _ = model(imgs, me, mp_)
                    loss = AllReduce.apply(loss)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(model.named_parameters())
                optimizer.zero_grad()

                return float(loss), _new_lr, _new_wd, grad_stats

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, mask_ratio, etime)

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '(%.1f ms)'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024. ** 2,
                           time_meter.avg)
                    )
                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                            % (epoch + 1, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max)
                        )

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch + 1)


if __name__ == '__main__':
    main()
