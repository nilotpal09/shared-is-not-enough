# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MAE model: lightweight decoder on top of the existing VisionTransformer encoder.
# Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (arXiv 2111.06377)

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vision_transformer import Block, get_2d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder that reconstructs masked patches from encoder tokens."""

    def __init__(
        self,
        num_patches,
        encoder_embed_dim,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        patch_size=16,
        in_chans=3,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Project encoder dim -> decoder dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learnable mask token (position-agnostic; positional info added separately)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sincos positional embedding for the full patch grid
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            decoder_embed_dim, int(num_patches ** 0.5), cls_token=False
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Predict pixel values for each patch position
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder_out, masks_enc, masks_pred):
        """
        encoder_out : (B, N_visible, D_encoder)
        masks_enc   : list of one (B, N_visible) int tensor  -- visible patch indices
        masks_pred  : list of one (B, N_masked)  int tensor  -- masked patch indices
        returns     : (B, N_masked, patch_size^2 * in_chans)
        """
        B = encoder_out.shape[0]
        N_total = self.num_patches
        D = self.decoder_embed.out_features

        # Project visible tokens to decoder dimension
        x_vis = self.decoder_embed(encoder_out)  # (B, N_vis, D)

        # Build full-grid sequence: start from mask tokens, scatter visible tokens in.
        # mask_token is an nn.Parameter (fp32); under bf16 autocast x_vis is bf16, so
        # cast x to match x_vis to keep scatter happy.
        x = self.mask_token.expand(B, N_total, D).clone().to(x_vis.dtype)
        ids_vis = masks_enc[0]  # (B, N_vis)
        x.scatter_(1, ids_vis.unsqueeze(-1).expand(-1, -1, D), x_vis)

        # Add sincos positional embeddings to every position (visible + masked)
        x = x + self.decoder_pos_embed.to(x.dtype)

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Pixel prediction head
        x = self.decoder_pred(x)  # (B, N_total, patch_size^2 * in_chans)

        # Return predictions only at masked positions
        ids_masked = masks_pred[0]  # (B, N_masked)
        patch_dim = x.shape[-1]
        pred_masked = x.gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, patch_dim))
        return pred_masked  # (B, N_masked, patch_dim)


class MAEModel(nn.Module):
    """Full MAE: VisionTransformer encoder + MAEDecoder + reconstruction loss."""

    def __init__(self, encoder, decoder, patch_size, in_chans, normalize_pixels=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.normalize_pixels = normalize_pixels

    def patchify(self, imgs):
        """(B, C, H, W) -> (B, N, patch_size^2 * C) matching PatchEmbed's raster order."""
        B, C, H, W = imgs.shape
        p = self.patch_size
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def forward(self, imgs, masks_enc, masks_pred):
        # Encode: only visible patches pass through the transformer
        latent = self.encoder(imgs, masks_enc)  # (B, N_vis, D_enc)

        # Decode: reconstruct masked patches in pixel space
        pred = self.decoder(latent, masks_enc, masks_pred)  # (B, N_masked, patch_dim)

        loss = self._loss(imgs, pred, masks_pred)
        return loss, pred

    def _loss(self, imgs, pred, masks_pred):
        target = self.patchify(imgs)  # (B, N_total, patch_dim)

        if self.normalize_pixels:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        ids_masked = masks_pred[0]  # (B, N_masked)
        patch_dim = target.shape[-1]
        target_masked = target.gather(
            1, ids_masked.unsqueeze(-1).expand(-1, -1, patch_dim)
        )
        return F.mse_loss(pred, target_masked)
