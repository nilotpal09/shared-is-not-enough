# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
from functools import partial

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.mae.model import MAEDecoder, MAEModel
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_mae_checkpoint(device, r_path, model, opt, scaler):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        msg = model.load_state_dict(checkpoint['model'])
        logger.info(f'loaded MAE model from epoch {epoch} with msg: {msg}')

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return model, opt, scaler, epoch


def init_mae_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    decoder_depth=8,
    decoder_embed_dim=512,
    decoder_num_heads=16,
    normalize_pixels=True,
    in_chans=3,
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        in_chans=in_chans,
    )
    decoder = MAEDecoder(
        num_patches=encoder.patch_embed.num_patches,
        encoder_embed_dim=encoder.embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        patch_size=patch_size,
        in_chans=in_chans,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = MAEModel(encoder, decoder, patch_size, in_chans, normalize_pixels)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)
    for m in decoder.modules():
        init_weights(m)

    model.to(device)
    logger.info(encoder)
    logger.info(decoder)
    return model


def init_mae_opt(
    model,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.0,
):
    param_groups = [
        {
            'params': (
                p for n, p in model.named_parameters()
                if ('bias' not in n) and (len(p.shape) != 1)
            )
        },
        {
            'params': (
                p for n, p in model.named_parameters()
                if ('bias' in n) or (len(p.shape) == 1)
            ),
            'WD_exclude': True,
            'weight_decay': 0,
        },
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
