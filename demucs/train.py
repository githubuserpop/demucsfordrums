#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training script entry point"""

import logging
import os
import sys
from pathlib import Path

from dora import hydra_main, get_xp
import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
import torchaudio
from torch.utils.data import ConcatDataset

from . import distrib
from .wav import get_wav_datasets, get_musdb_wav_datasets
from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .repitch import RepitchedWrapper
from .solver import Solver
from .states import capture_init
from .utils import random_subset
from .drum_datasets import get_drum_datasets, DrumDataset

logger = logging.getLogger(__name__)

TRAINING_FOLDERS = [
    '/DATA/Training Data/organized drums',
    '/DATA/Training Data/beat stems',
    '/DATA/Training Data/song drum stems'
]

class TorchHDemucsWrapper(nn.Module):
    """Wrapper around torchaudio HDemucs implementation to provide the proper metadata for model evaluation."""
    
    @capture_init
    def __init__(self, **kwargs):
        super().__init__()
        from torchaudio.models import HDemucs as TorchHDemucs
        self.samplerate = kwargs.pop('samplerate')
        self.segment = kwargs.pop('segment')
        self.sources = kwargs['sources']
        self.torch_hdemucs = TorchHDemucs(**kwargs)

    def forward(self, mix):
        return self.torch_hdemucs(mix)


def get_model(args):
    """Initializes the model based on specified configuration."""
    extra = {
        'sources': list(args.dset.sources),
        'audio_channels': args.dset.channels,
        'samplerate': args.dset.samplerate,
        'segment': args.model_segment or 4 * args.dset.segment,
    }
    klass = {
        'demucs': Demucs,
        'hdemucs': HDemucs,
        'htdemucs': HTDemucs,
        'torch_hdemucs': TorchHDemucsWrapper,
    }.get(args.model)
    if klass is None:
        raise ValueError(f"Model type '{args.model}' is not recognized.")
    
    kw = OmegaConf.to_container(getattr(args, args.model), resolve=True)
    return klass(**extra, **kw)


def get_optimizer(model, args):
    """Sets up optimizer based on configuration."""
    seen_params, other_params, groups = set(), [], []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            assert set(group["params"]).isdisjoint(seen_params)
            seen_params |= set(group["params"])
            groups.append(group)
    other_params = [param for param in model.parameters() if param not in seen_params]
    groups.insert(0, {"params": other_params})

    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    optimizer_cls = optimizers.get(args.optim.optim.lower())
    if optimizer_cls is None:
        raise ValueError(f"Invalid optimizer type '{args.optim.optim}'")
    return optimizer_cls(groups, lr=args.optim.lr, betas=(args.optim.momentum, args.optim.beta2), weight_decay=args.optim.weight_decay)


def get_datasets(args):
    """Creates train and validation datasets."""
    if args.dset.backend:
        torchaudio.set_audio_backend(args.dset.backend)

    # If using drum-specific dataset
    if args.model == 'hdemucs' and hasattr(args.dset, 'use_drums') and args.dset.use_drums:
        return get_drum_datasets(args)

    # Original dataset loading logic for other cases
    train_set, valid_set = (get_musdb_wav_datasets(args.dset) if args.dset.use_musdb else ([], []))
    if args.dset.wav:
        extra_train_set, extra_valid_set = get_wav_datasets(args.dset)
        train_set, valid_set = ConcatDataset([train_set, extra_train_set]), ConcatDataset([valid_set, extra_valid_set])

    if args.dset.wav2:
        extra_train_set, extra_valid_set = get_wav_datasets(args.dset, "wav2")
        reps = max(1, round(len(extra_train_set) / len(train_set) * (1 / args.dset.wav2_weight - 1))) if args.dset.wav2_weight else 1
        train_set = ConcatDataset([train_set] * reps + [extra_train_set])
        if args.dset.wav2_valid and args.dset.wav2_weight:
            valid_set = ConcatDataset([valid_set, random_subset(extra_valid_set, int(round(args.dset.wav2_weight * len(valid_set) / (1 - args.dset.wav2_weight))))])

    if args.dset.valid_samples:
        valid_set = random_subset(valid_set, args.dset.valid_samples)
    return train_set, valid_set


def get_drum_datasets(args):
    """Creates train and validation datasets for drum separation."""
    sample_dirs = [
        '/DATA/Training Data/organized drums'
    ]
    
    stem_dirs = [
        '/DATA/Training Data/song drum stems',
        '/DATA/Training Data/beat stems'
    ]
    
    train_dataset = DrumDataset(
        sample_dirs=sample_dirs,
        stem_dirs=stem_dirs,
        sources=args.dset.sources,
        sample_rate=args.dset.samplerate,
        segment_duration=args.dset.segment,
        channels=args.dset.channels,
        normalize=args.dset.normalize,
        sample_ratio=args.dset.sample_ratio
    )
    
    # Create a smaller validation set (10% of training data)
    valid_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_set, valid_set = torch.utils.data.random_split(
        train_dataset, 
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    return train_set, valid_set


def get_solver(args, model_only=False):
    distrib.init()
    torch.manual_seed(args.seed)
    model = get_model(args)
    
    if args.misc.show:
        logger.info(model)
        logger.info(f'Size: {sum(p.numel() for p in model.parameters()) * 4 / 2**20:.1f} MB')
        if hasattr(model, 'valid_length'):
            logger.info(f'Field: {model.valid_length(1) / args.dset.samplerate * 1000:.1f} ms')
        sys.exit(0)

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    optimizer = get_optimizer(model, args)
    args.batch_size //= distrib.world_size

    if model_only:
        return Solver(None, model, optimizer, args)

    train_set, valid_set = get_datasets(args)
    train_loader = distrib.loader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.misc.num_workers, drop_last=True)
    valid_loader = distrib.loader(valid_set, batch_size=1 if args.dset.full_cv else args.batch_size, shuffle=False, num_workers=args.misc.num_workers, drop_last=not args.dset.full_cv)
    loaders = {"train": train_loader, "valid": valid_loader}

    logger.info(f"Train/valid set sizes: {len(train_set)}, {len(valid_set)}")
    return Solver(loaders, model, optimizer, args)


def load_custom_data(training_folders, batch_size):
    """
    Function to set up the dataloader for training data across the specified folders
    with variable sample rates.
    """
    print("Loading data...")
    datasets = []
    for folder in training_folders:
        if os.path.exists(folder):
            print(f"Loading data from: {folder}")
            # Initialize the dataset for each folder
            dataset = Dataset(
                root=folder,
                streams=["drums"],  # Focus on drums
                channels=2,         # Stereo audio
                duration=10,        # Segment duration in seconds
                samplerate=None,    # Use the sample rate of the files (no resampling)
                stride=5,           # Overlap between segments
                augment=True        # Apply data augmentation
            )
            datasets.append(dataset)
        else:
            print(f"Folder not found: {folder}")
    
    if datasets:
        print("Combining datasets...")
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return dataloader
    else:
        raise RuntimeError("No datasets found in the specified folders.")


@hydra_main(config_path="../conf", config_name="config", version_base="1.1")
def main(args):
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val:
            setattr(args.dset, attr, hydra.utils.to_absolute_path(val))

    os.environ["OMP_NUM_THREADS"], os.environ["MKL_NUM_THREADS"] = "1", "1"
    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"For logs, checkpoints, and samples, check {os.getcwd()}")
    logger.debug(args)
    logger.debug(get_xp().cfg)

    solver = get_solver(args)
    try:
        solver.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current checkpoint.")
        solver._serialize(epoch=solver.history[-1] if solver.history else 0)


if '_DORA_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_DORA_TEST_PATH'])

if __name__ == "__main__":
    main()