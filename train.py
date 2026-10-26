# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from I-JEPA's main.py: one entry point for all four objectives.

"""Train one encoder.

    python train.py --config configs/ijepa/ijepa_shared_default.yml --seed 3

The objective comes from the config's meta.objective (ijepa, contrastive for
SimCLR*, byol, mae). Checkpoints, the loss log and a copy of the config go to
<out>/<config name>_seed<seed>/.
"""

import argparse
import multiprocessing as mp
import os
import pprint
import sys

import yaml

LOOPS = {
    'ijepa': 'src.train',
    'contrastive': 'src.train_contrastive',
    'byol': 'src.train_byol',
    'mae': 'src.mae.train',
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='config file to load')
parser.add_argument('--seed', type=int, default=None, help='training seed (overrides meta.seed)')
parser.add_argument('--out', type=str, default='runs', help='parent folder for the run folder')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, params, world_size, devices, run_folder):
    import importlib
    import logging
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    app_main = importlib.import_module(LOOPS[params['meta']['objective']]).main
    from src.utils.distributed import init_distributed
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params, run_folder=run_folder)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    if params['meta'].get('objective') not in LOOPS:
        raise ValueError(f"meta.objective must be one of {sorted(LOOPS)}")
    if args.seed is not None:
        params['meta']['seed'] = args.seed
    name = os.path.splitext(os.path.basename(args.config))[0]
    run_folder = os.path.join(args.out, f"{name}_seed{params['meta'].get('seed', 1)}")

    mp.set_start_method('spawn', force=True)
    procs = []
    num_gpus = len(args.devices)
    for rank in range(num_gpus):
        p = mp.Process(target=process_main,
                       args=(rank, params, num_gpus, args.devices, run_folder))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    sys.exit(max(p.exitcode for p in procs))
