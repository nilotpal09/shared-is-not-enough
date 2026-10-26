# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def _pick_port():
    """Pick a port that's unlikely to collide on a shared node.

    Priority: MASTER_PORT env override → PBS_JOBID-derived (deterministic across
    ranks of the same job) → random fallback.
    """
    env = os.environ.get('MASTER_PORT')
    if env:
        return int(env)
    jid = os.environ.get('PBS_JOBID') or os.environ.get('SLURM_JOB_ID')
    if jid:
        return 30000 + (abs(hash(jid)) % 20000)
    return random.randint(30000, 50000)


def init_distributed(port=None, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank

    if port is None:
        port = _pick_port()
    os.environ['MASTER_PORT'] = str(port)
    try:
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        logger.warning(f'nccl init failed on port {port} ({e}), retrying with gloo...')
        try:
            torch.distributed.init_process_group(
                backend='gloo',
                world_size=world_size,
                rank=rank)
        except Exception as e2:
            # Hard failure — DDP will explode if we don't surface this.
            raise RuntimeError(
                f'distributed init failed on port {port}: nccl={e!r}; gloo={e2!r}'
            ) from e2

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
